#pragma once

#include <sycl/sycl.hpp>
#include "SimulationUtils.hpp"
#include "GridField.hpp"
#include "Field/ElectromagneticField.hpp"
#include "Chunk.hpp"

/**
 * @class BorisPusherStrategy
 * @brief Relativistic Boris pusher strategy for electromagnetic field particle acceleration.
 * 
 * Implements the standard leapfrog-based Boris algorithm, separating electric field 
 * half-acceleration steps from full magnetic field rotations.
 */
// using FieldPolicy = ElectromagneticField; // BorishPusherStrategy always operate in ElectromagneticField
template <typename FieldPolicy>
class BorisPusherStrategy {
private:
    FieldPolicy field_policy_;

public:
    /// @brief Default constructor for BorisPusherStrategy.
    BorisPusherStrategy() : field_policy_(FieldPolicy()) {}

    /**
     * @brief Executes the Boris pusher algorithm for a single particle chunk on the device.
     * 
     * @param local_view Structure containing pointers to the SoA particle data buffers (`px`, `vx`, etc.).
     * @param dt Timestep size [s].
     * @param q_over_m Charge-to-mass ratio of the species [C/kg].
     * @param grid Reference to the grid and field data manager.
     * @param cgh SYCL command group handler for kernel submission.
     */
    void push(const ChunkSoAView &local_view, float dt, float q_over_m, GridField &grid, sycl::handler &cgh) const {
        // Możemy przekazać odpowiednie akcesory z siatki, jeśli GridField je udostępnia
        // Na ten moment używamy uproszczonych wartości lub gotowych struktur na GPU
        auto field_policy = field_policy_;
        auto grid_device_view = grid.get_device_view();

        cgh.parallel_for(sycl::range<1>(local_view.count), [=, dt = static_cast<double>(dt), q_over_m = static_cast<double>(q_over_m)](sycl::id<1> idx) {
            size_t i = idx[0];

            // 1. Pobranie pozycji i prędkości cząstki
            Vector3d pos{static_cast<double>(local_view.px[i]), static_cast<double>(local_view.py[i]), static_cast<double>(local_view.pz[i])};
            Vector3d vel{static_cast<double>(local_view.vx[i]), static_cast<double>(local_view.vy[i]), static_cast<double>(local_view.vz[i])};

            // 2. Pobranie pól E i B z siatki (na razie dummy values / stałe pola testowe)
            Vector3 E_f, B_f;
            field_policy.device_sample(Vector3{static_cast<float>(pos.x), static_cast<float>(pos.y), static_cast<float>(pos.z)}, E_f, B_f);
            
            Vector3d E{static_cast<double>(E_f.x), static_cast<double>(E_f.y), static_cast<double>(E_f.z)};
            Vector3d B{static_cast<double>(B_f.x), static_cast<double>(B_f.y), static_cast<double>(B_f.z)};

            constexpr double c = Constants::c; // Prędkość światła [m/s]
            constexpr double c2 = c * c;
            constexpr double inv_c2 = 1.0 / c2;

            // --- PRZEJŚCIE NA PRĘDKOŚĆ WŁAŚCIWĄ (PROPER VELOCITY) u = gamma * v ---
            double vv = Vector3d::dot(vel, vel);
            double vv_safe = sycl::fmin(vv, c2 * 0.999999);
            double gamma_init = 1.0 / sycl::sqrt(sycl::fmax(1e-15, 1.0 - vv_safe * inv_c2));
            Vector3d u = Vector3d::scale(vel, gamma_init);

            // --- ADAPTYWNE PODKROKI ---
            int sub_steps = device_compute_boris_algorithm_sub_steps(
                Vector3{static_cast<float>(vel.x), static_cast<float>(vel.y), static_cast<float>(vel.z)}, 
                E_f, B_f, static_cast<float>(dt), static_cast<float>(q_over_m)
            );
            double sub_dt = dt / static_cast<double>(sub_steps);

            // --- POCZĄTEK ALGORITMU BORISA (Relatywistycznego) ---
            for(auto step = 0; step < sub_steps; ++step) {            
                // A. PIERWSZY PÓŁ-KROK ELEKTRYCZNY Z POPRAWKĄ RELATYWISTYCZNĄ
                double uu = Vector3d::dot(u, u);
                double gamma = sycl::sqrt(1.0 + uu * inv_c2);
                double q_prim = 0.5 * sub_dt * q_over_m; // half of time step times q over m
                
                // Poprawka podłużna: E - (u · E) * u / (gamma^2 * c^2)
                double uE = Vector3d::dot(u, E);
                double factor_long = uE / (gamma * gamma * c2);
                Vector3d u_factor_long = Vector3d::scale(u, factor_long);
                Vector3d E_eff1 = Vector3d::sub(E, u_factor_long);
                
                Vector3d impulse1 = Vector3d::scale(E_eff1, q_prim);
                Vector3d u_minus = Vector3d::add(u, impulse1);

                // B. ROTACJA MAGNETYCZNA BORISA
                double uu_minus = Vector3d::dot(u_minus, u_minus);
                double gamma_minus = sycl::sqrt(1.0 + uu_minus * inv_c2);
                
                double t_factor = q_prim / gamma_minus;
                Vector3d h = Vector3d::scale(B, t_factor);

                double hh = Vector3d::dot(h, h);
                double s_denom = 2.0 / (1.0 + hh);
                Vector3d s = Vector3d::scale(h, s_denom);

                Vector3d u_minus_cross_h = Vector3d::cross(u_minus, h);
                Vector3d u_prim = Vector3d::add(u_minus, u_minus_cross_h);

                Vector3d u_prim_cross_s = Vector3d::cross(u_prim, s);
                Vector3d u_plus = Vector3d::add(u_minus, u_prim_cross_s);

                // C. DRUGI PÓŁ-KROK ELEKTRYCZNY (symetryczny z u_plus)
                double uu_plus = Vector3d::dot(u_plus, u_plus);
                double gamma_plus = sycl::sqrt(1.0 + uu_plus * inv_c2);
                
                double uE_plus = Vector3d::dot(u_plus, E);
                double factor_long_plus = uE_plus / (gamma_plus * gamma_plus * c2);
                Vector3d u_factor_long_plus = Vector3d::scale(u_plus, factor_long_plus);
                Vector3d E_eff2 = Vector3d::sub(E, u_factor_long_plus);
                
                Vector3d impulse2 = Vector3d::scale(E_eff2, q_prim);
                u = Vector3d::add(u_plus, impulse2);

                // D. AKTUALIZACJA POZYCJI W PODKROKU
                double uu_new = Vector3d::dot(u, u);
                double gamma_new = sycl::sqrt(1.0 + uu_new * inv_c2);
                Vector3d v_sub = Vector3d::scale(u, 1.0 / gamma_new);

                pos = Vector3d::add(pos, Vector3d::scale(v_sub, sub_dt));
            }
            // --- OSTATECZNE DOMKNIĘCIE KROKU I OCHRONA PRZED C ---
            double uu_final = Vector3d::dot(u, u);
            double gamma_final = sycl::sqrt(1.0 + uu_final * inv_c2);
            vel = Vector3d::scale(u, 1.0 / gamma_final);

            // --- KONIEC ALGORITMU BORISA ---

            bool is_out = grid_device_view.device_out_of_boundary_check(
                static_cast<float>(pos.x), static_cast<float>(pos.y), static_cast<float>(pos.z)
            );
            if(is_out){
                local_view.device_mark_for_removal(i);
            }

            // Zapis zaktualizowanych danych z powrotem do buforów SoA
            local_view.px[i] = static_cast<float>(pos.x);
            local_view.py[i] = static_cast<float>(pos.y);
            local_view.pz[i] = static_cast<float>(pos.z);

            local_view.vx[i] = static_cast<float>(vel.x);
            local_view.vy[i] = static_cast<float>(vel.y);
            local_view.vz[i] = static_cast<float>(vel.z);
        });
    }
    // void push(const ChunkSoAView &local_view, float dt, float q_over_m, GridField &grid, sycl::handler &cgh) const {
    //     // Możemy przekazać odpowiednie akcesory z siatki, jeśli GridField je udostępnia
    //     // Na ten moment używamy uproszczonych wartości lub gotowych struktur na GPU
    //     auto field_policy = field_policy_;
    //     auto grid_device_view = grid.get_device_view();

    //     cgh.parallel_for(sycl::range<1>(local_view.count), [=, dt = dt, q_over_m = q_over_m](sycl::id<1> idx) {
    //         size_t i = idx[0];

    //         // 1. Pobranie pozycji i prędkości cząstki
    //         Vector3 pos{local_view.px[i], local_view.py[i], local_view.pz[i]};
    //         Vector3 vel{local_view.vx[i], local_view.vy[i], local_view.vz[i]};

    //         // 2. Pobranie pól E i B z siatki (na razie dummy values / stałe pola testowe)
    //         Vector3 E, B;
    //         field_policy.device_sample(pos, E, B);

    //         constexpr float c = static_cast<float>(Constants::c); // Prędkość światła [m/s]
    //         constexpr float c2 = c * c;
    //         constexpr float inv_c2 = 1.0f / c2;

    //         // --- PRZEJŚCIE NA PRĘDKOŚĆ WŁAŚCIWĄ (PROPER VELOCITY) u = gamma * v ---
    //         float vv = Vector3::dot(vel, vel);
    //         float vv_safe = sycl::fmin(vv, c2 * 0.999999f);
    //         float gamma_init = 1.0f / sycl::sqrt(1.0f - vv_safe * inv_c2);
    //         Vector3 u = Vector3::scale(vel, gamma_init);

    //         // --- ADAPTYWNE PODKROKI ---
    //         int sub_steps = device_compute_boris_algorithm_sub_steps(vel, E, B, dt, q_over_m);
    //         float sub_dt = dt / static_cast<float>(sub_steps);

    //         // --- POCZĄTEK ALGORITMU BORISA (Relatywistycznego) ---
    //         for(auto step = 0; step < sub_steps; ++step) {             
    //             // A. PIERWSZY PÓŁ-KROK ELEKTRYCZNY Z POPRAWKĄ RELATYWISTYCZNĄ
    //             float uu = Vector3::dot(u, u);
    //             float gamma = sycl::sqrt(1.0f + uu * inv_c2);
    //             float q_prim = 0.5f * sub_dt * q_over_m; // half of time step times q over m
                
    //             // Poprawka podłużna: E - (u · E) * u / (gamma^2 * c^2)
    //             float uE = Vector3::dot(u, E);
    //             float factor_long = uE / (gamma * gamma * c2);
    //             Vector3 u_factor_long = Vector3::scale(u, factor_long);
    //             Vector3 E_eff1 = Vector3::sub(E, u_factor_long);
                
    //             Vector3 impulse1 = Vector3::scale(E_eff1, q_prim);
    //             Vector3 u_minus = Vector3::add(u, impulse1);

    //             // B. ROTACJA MAGNETYCZNA BORISA
    //             float uu_minus = Vector3::dot(u_minus, u_minus);
    //             float gamma_minus = sycl::sqrt(1.0f + uu_minus * inv_c2);
                
    //             float t_factor = q_prim / gamma_minus;
    //             Vector3 h = Vector3::scale(B, t_factor);

    //             float hh = Vector3::dot(h, h);
    //             float s_denom = 2.0f / (1.0f + hh);
    //             Vector3 s = Vector3::scale(h, s_denom);

    //             Vector3 u_minus_cross_h = Vector3::cross(u_minus, h);
    //             Vector3 u_prim = Vector3::add(u_minus, u_minus_cross_h);

    //             Vector3 u_prim_cross_s = Vector3::cross(u_prim, s);
    //             Vector3 u_plus = Vector3::add(u_minus, u_prim_cross_s);

    //             // C. DRUGI PÓŁ-KROK ELEKTRYCZNY (symetryczny z u_plus)
    //             float uu_plus = Vector3::dot(u_plus, u_plus);
    //             float gamma_plus = sycl::sqrt(1.0f + uu_plus * inv_c2);
                
    //             float uE_plus = Vector3::dot(u_plus, E);
    //             float factor_long_plus = uE_plus / (gamma_plus * gamma_plus * c2);
    //             Vector3 u_factor_long_plus = Vector3::scale(u_plus, factor_long_plus);
    //             Vector3 E_eff2 = Vector3::sub(E, u_factor_long_plus);
                
    //             Vector3 impulse2 = Vector3::scale(E_eff2, q_prim);
    //             u = Vector3::add(u_plus, impulse2);

    //             // D. AKTUALIZACJA POZYCJI W PODKROKU
    //             float uu_new = Vector3::dot(u, u);
    //             float gamma_new = sycl::sqrt(1.0f + uu_new * inv_c2);
    //             Vector3 v_sub = Vector3::scale(u, 1.0f / gamma_new);

    //             pos = Vector3::add(pos, Vector3::scale(v_sub, sub_dt));
    //         }
    //         // --- OSTATECZNE DOMKNIĘCIE KROKU I OCHRONA PRZED C ---
    //         float uu_final = Vector3::dot(u, u);
    //         float gamma_final = sycl::sqrt(1.0f + uu_final * inv_c2);
    //         vel = Vector3::scale(u, 1.0f / gamma_final);

    //         // --- KONIEC ALGORITMU BORISA ---

    //         bool is_out = grid_device_view.device_out_of_boundary_check(pos.x, pos.y, pos.z);
    //         if(is_out){
    //             local_view.device_mark_for_removal(i);
    //         }

    //         // Zapis zaktualizowanych danych z powrotem do buforów SoA
    //         local_view.px[i] = pos.x;
    //         local_view.py[i] = pos.y;
    //         local_view.pz[i] = pos.z;

    //         local_view.vx[i] = vel.x;
    //         local_view.vy[i] = vel.y;
    //         local_view.vz[i] = vel.z;
    //     });
    // }

    /**
     * @brief Retrieves a constant reference to the underlying field policy.
     * @return Const reference to the FieldPolicy object.
     */
    const FieldPolicy& get_field_policy() const {
        return field_policy_;
    }

    /**
     * @brief Automatically computes the required number of dynamic sub-steps for the Boris pusher
     *        based on magnetic cyclotron frequency and electric acceleration thresholds.
     */
    static inline int device_compute_boris_algorithm_sub_steps(Vector3 vel, Vector3 E, Vector3 B, float dt, float q_over_m) {
        constexpr float c = static_cast<float>(Constants::c);
        constexpr float c2 = c * c;
        constexpr float inv_c2 = 1.0f / c2;

        float B_mag = sycl::sqrt(Vector3::dot(B, B));
        float E_mag = sycl::sqrt(Vector3::dot(E, E));
        float speed_sq = Vector3::dot(vel, vel);

        // Czynnik gamma dla bieżącego stanu
        float gamma = 1.0f / sycl::sqrt(sycl::fmax(1e-7f, 1.0f - speed_sq * inv_c2));

        // 1. Kryterium magnetyczne: max 45 stopni (pi/4 rad) na podkrok
        float omega_c = sycl::fabs(q_over_m) * B_mag / gamma;
        constexpr float max_rotation_angle = static_cast<float>(Constants::PI/6); // 30 stopni w radianach (pi / 6)
        int sub_steps_mag = static_cast<int>(sycl::ceil((omega_c * dt) / max_rotation_angle));

        // 2. Kryterium elektryczne (relatywistyczne): ograniczenie zmiany pędu właściwego u na podkrok do max 20%
        // Ponieważ du = (q/m) * E * dt, sprawdzamy ułamek względem obecnego pędu u = gamma * v (lub skali c)
        float u_mag = gamma * sycl::sqrt(speed_sq);
        float acceleration_u = sycl::fabs(q_over_m) * E_mag;
        
        // Jeśli cząstka stoi w spoczynku (u_mag = 0), odnosimy kryterium do c, żeby uniknąć dzielenia przez zero
        float reference_scale = sycl::fmax(u_mag, 0.1f * c);
        int sub_steps_elec = static_cast<int>(sycl::ceil((acceleration_u * dt) / (0.2f * reference_scale)));

        // 3. Wybór najbardziej rygorystycznego wymogu (zakres od 1 do 32 podkroków)
        int sub_steps = sycl::max(1, sycl::max(sub_steps_mag, sub_steps_elec));
        return sycl::min(sub_steps, 32);
    }
};