#pragma once

#include <sycl/sycl.hpp>
#include <concepts>
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

        cgh.parallel_for(sycl::range<1>(local_view.count), [=, dt = dt, q_over_m = q_over_m](sycl::id<1> idx) {
            size_t i = idx[0];

            // Vector3 pos{local_view.px[i], local_view.py[i], local_view.pz[i]};
            // Vector3 vel{local_view.vx[i], local_view.vy[i], local_view.vz[i]};
            // device_boris_algorithm<float>(pos, vel, dt, q_over_m, field_policy);

            // bool is_out = grid_device_view.device_out_of_boundary_check(pos.x, pos.y, pos.z);
            // if(is_out){ local_view.device_mark_for_removal(i); }

            // // Zapis zaktualizowanych danych z powrotem do buforów SoA
            // local_view.px[i] = pos.x;
            // local_view.py[i] = pos.y;
            // local_view.pz[i] = pos.z;

            // local_view.vx[i] = vel.x;
            // local_view.vy[i] = vel.y;
            // local_view.vz[i] = vel.z;

            Vec3<double> pos = {
                static_cast<double>(local_view.px[i]),
                static_cast<double>(local_view.py[i]),
                static_cast<double>(local_view.pz[i])
            };
            Vec3<double> vel = {
                static_cast<double>(local_view.vx[i]),
                static_cast<double>(local_view.vy[i]),
                static_cast<double>(local_view.vz[i])
            };
            device_boris_algorithm<double>(pos, vel, static_cast<double>(dt), static_cast<double>(q_over_m), field_policy);

            bool is_out = grid_device_view.device_out_of_boundary_check(pos.x, pos.y, pos.z);
            if(is_out){ local_view.device_mark_for_removal(i); }

            local_view.px[i] = static_cast<float>(pos.x);
            local_view.py[i] = static_cast<float>(pos.y);
            local_view.pz[i] = static_cast<float>(pos.z);

            local_view.vx[i] = static_cast<float>(vel.x);
            local_view.vy[i] = static_cast<float>(vel.y);
            local_view.vz[i] = static_cast<float>(vel.z);
        });
    }

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
    template <typename Float>
    requires std::floating_point<Float>
    static inline int device_compute_boris_algorithm_sub_steps(Vec3<Float> vel, Vec3<Float> E, Vec3<Float> B, Float dt, Float q_over_m) {
        using Vec3 = Vec3<Float>;
        constexpr Float c = static_cast<Float>(Constants::c);
        constexpr Float c2 = c * c;
        constexpr Float inv_c2 = 1.0f / c2;

        Float B_mag = sycl::sqrt(Vec3::dot(B, B));
        Float E_mag = sycl::sqrt(Vec3::dot(E, E));
        Float speed_sq = Vec3::dot(vel, vel);

        // Czynnik gamma dla bieżącego stanu
        Float gamma = static_cast<Float>(1.0) / sycl::sqrt(sycl::fmax(static_cast<Float>(1e-7), static_cast<Float>(1.0) - speed_sq * inv_c2));

        // 1. Kryterium magnetyczne: max 30 stopni (pi/6 rad) na podkrok
        Float omega_c = sycl::fabs(q_over_m) * B_mag / gamma;
        constexpr Float max_rotation_angle = static_cast<Float>(Constants::PI/6); // 30 stopni w radianach (pi / 6)
        int sub_steps_mag = static_cast<int>(sycl::ceil((omega_c * dt) / max_rotation_angle));

        // 2. Kryterium elektryczne (relatywistyczne): ograniczenie zmiany pędu właściwego u na podkrok do max 20%
        // Ponieważ du = (q/m) * E * dt, sprawdzamy ułamek względem obecnego pędu u = gamma * v (lub skali c)
        Float u_mag = gamma * sycl::sqrt(speed_sq);
        Float acceleration_u = sycl::fabs(q_over_m) * E_mag;
        
        // Jeśli cząstka stoi w spoczynku (u_mag = 0), odnosimy kryterium do c, żeby uniknąć dzielenia przez zero
        Float reference_scale = sycl::fmax(u_mag, static_cast<Float>(0.1) * c);
        int sub_steps_elec = static_cast<int>(sycl::ceil((acceleration_u * dt) / (static_cast<Float>(0.2) * reference_scale)));

        // 3. Wybór najbardziej rygorystycznego wymogu (zakres od 1 do 32 podkroków)
        int sub_steps = sycl::max(1, sycl::max(sub_steps_mag, sub_steps_elec));
        return sycl::min(sub_steps, 32);
    }

    template <typename Float>
    requires std::floating_point<Float>
    static inline void device_boris_algorithm(Vec3<Float> &pos, Vec3<Float> &vel, Float dt, Float q_over_m, const FieldPolicy &field_policy) {
        // Pobranie pól E i B z siatki (na razie dummy values / stałe pola testowe)
        using Vec3 = Vec3<Float>;
        
        Vec3 E, B;
        field_policy.template device_sample<Float>(pos, E, B);

        constexpr Float c = static_cast<Float>(Constants::c); // Prędkość światła [m/s]
        constexpr Float c2 = c * c;
        constexpr Float inv_c2 = static_cast<Float>(1.0) / c2;

        // --- PRZEJŚCIE NA PRĘDKOŚĆ WŁAŚCIWĄ (PROPER VELOCITY) u = gamma * v ---
        Float vv = Vec3::dot(vel, vel);
        Float vv_safe = sycl::fmin(vv, c2 * static_cast<Float>(0.999999));
        Float gamma_init = static_cast<Float>(1.0) / sycl::sqrt(sycl::fmax(static_cast<Float>(1e-15), static_cast<Float>(1.0) - vv_safe * inv_c2));

        Vec3 u = Vec3::scale(vel, gamma_init);

        // --- ADAPTYWNE PODKROKI ---
        int sub_steps = device_compute_boris_algorithm_sub_steps<Float>(vel, E, B, dt, q_over_m);
        Float sub_dt = dt / static_cast<Float>(sub_steps);

        // --- POCZĄTEK ALGORITMU BORISA (Relatywistycznego) ---
        for (auto step = 0; step < sub_steps; ++step)
        {
            field_policy.template device_sample<Float>(pos, E, B);

            // A. PIERWSZY PÓŁ-KROK ELEKTRYCZNY Z POPRAWKĄ RELATYWISTYCZNĄ
            Float uu_old = Vec3::dot(u, u);
            Float gamma_old = sycl::sqrt(static_cast<Float>(1.0) + uu_old * inv_c2);
            Vec3 v_old = Vec3::scale(u, static_cast<Float>(1.0) / gamma_old);

            // Poprawka podłużna: u^- = u + q'E
            Float q_prim = static_cast<Float>(0.5) * sub_dt * q_over_m; // half of time step times q over m
            Vec3 impulse1 = Vec3::scale(E, q_prim);
            Vec3 u_minus = Vec3::add(u, impulse1);

            // B. ROTACJA MAGNETYCZNA BORISA
            Float uu_minus = Vec3::dot(u_minus, u_minus);
            Float gamma_minus = sycl::sqrt(static_cast<Float>(1.0) + uu_minus * inv_c2);

            Float t_factor = q_prim / gamma_minus;
            Vec3 h = Vec3::scale(B, t_factor);

            Float hh = Vec3::dot(h, h);
            Float s_denom = static_cast<Float>(2.0) / (static_cast<Float>(1.0) + hh);
            Vec3 s = Vec3::scale(h, s_denom);

            Vec3 u_minus_cross_h = Vec3::cross(u_minus, h);
            Vec3 u_prim = Vec3::add(u_minus, u_minus_cross_h);

            Vec3 u_prim_cross_s = Vec3::cross(u_prim, s);
            Vec3 u_plus = Vec3::add(u_minus, u_prim_cross_s);

            // C. DRUGI PÓŁ-KROK ELEKTRYCZNY (symetryczny z u_plus)
            Vec3 impulse2 = Vec3::scale(E, q_prim);
            u = Vec3::add(u_plus, impulse2);

            // D. AKTUALIZACJA POZYCJI W PODKROKU
            Float uu_new = Vec3::dot(u, u);
            Float gamma_new = sycl::sqrt(static_cast<Float>(1.0) + uu_new * inv_c2);
            Vec3 v_new = Vec3::scale(u, static_cast<Float>(1.0) / gamma_new);

            Vec3 v_avg = Vec3::scale(Vec3::add(v_old, v_new), static_cast<Float>(0.5));

            pos = Vec3::add(pos, Vec3::scale(v_avg, sub_dt));
        }

        // --- OSTATECZNE DOMKNIĘCIE KROKU I OCHRONA PRZED C ---
        Float uu_final = Vec3::dot(u, u);
        Float gamma_final = sycl::sqrt(static_cast<Float>(1.0) + uu_final * inv_c2);
        vel = Vec3::scale(u, static_cast<Float>(1.0) / gamma_final);

        // --- KONIEC ALGORITMU BORISA ---
    }
};