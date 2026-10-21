#pragma once

#include <sycl/sycl.hpp>
#include "SimulationUtils.hpp"
#include "GridField.hpp"
#include "Field/ElectromagneticField.hpp"

/**
 * @class BorisPusherStrategy
 * @brief Relativistic Boris pusher strategy for electromagnetic field particle acceleration.
 * 
 * Implements the standard leapfrog-based Boris algorithm, separating electric field 
 * half-acceleration steps from full magnetic field rotations.
 */
// template <typename FieldPolicy>
using FieldPolicy = ElectromagneticField; // BorishPusherStrategy always operate in ElectromagneticField
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
        auto grid_device_vew = grid.get_device_view();

        cgh.parallel_for(sycl::range<1>(local_view.count), [=, dt = dt, q_over_m = q_over_m](sycl::id<1> idx) {
            size_t i = idx[0];

            // 1. Pobranie pozycji i prędkości cząstki
            float x  = local_view.px[i];
            float y  = local_view.py[i];
            float z  = local_view.pz[i];
            
            float vx = local_view.vx[i];
            float vy = local_view.vy[i];
            float vz = local_view.vz[i];

            // 2. Pobranie pól E i B z siatki (na razie dummy values / stałe pola testowe)
            // W przyszłości: Vector3 E = grid.device_sample_electric_field(x, y, z);
            float Ex, Ey, Ez;
            float Bx, By, Bz; // Przykładowe pole magnetyczne w osi Z
            field_policy.device_sample(x, y, z, Ex, Ey, Ez, Bx, By, Bz);

            // --- POCZÁTEK ALGORITMU BORISA (Relatywistycznego) ---
            constexpr float c = static_cast<float>(Constants::c); // Prędkość światła [m/s]
            constexpr float c2 = c * c;

            float vv = vx * vx + vy * vy + vz * vz;
            
            // Czynnik relatywistyczny t = (q / m) * (dt / 2) / gamma
            float t = 0.0f;
            if (vv < c2) {
                float gamma_factor = sycl::sqrt(1.0f - vv / c2);
                t = (q_over_m) * (dt * 0.5f) * gamma_factor;
            }

            // Wektor w = t * E (uproszczony dla czystego pola E, pełny relatywistyczny uwzględnia (v x E))
            float wx = t * Ex;
            float wy = t * Ey;
            float wz = t * Ez;

            // Prędkość pośrednia u^- = v + w
            float ux = vx + wx;
            float uy = vy + wy;
            float uz = vz + wz;

            // Wektor h = t * B
            float hx = t * Bx;
            float hy = t * By;
            float hz = t * Bz;

            // s = 2h / (1 + h*h)
            float h2 = hx * hx + hy * hy + hz * hz;
            float s_denom = 2.0f / (1.0f + h2);
            float sx = hx * s_denom;
            float sy = hy * s_denom;
            float sz = hz * s_denom;

            // Rotacja: u' = u^- + (u^- x h)
            float u_prim_x = ux + (uy * hz - uz * hy);
            float u_prim_y = uy + (uz * hx - ux * hz);
            float u_prim_z = uz + (ux * hy - uy * hx);

            // Domknięcie rotacji: u^+ = u^- + (u' x s)
            float u_plus_x = ux + (u_prim_y * sz - u_prim_z * sy);
            float u_plus_y = uy + (u_prim_z * sx - u_prim_x * sz);
            float u_plus_z = uz + (u_prim_x * sy - u_prim_y * sx);

            // Nowa prędkość v = u^+ + w
            vx = u_plus_x + wx;
            vy = u_plus_y + wy;
            vz = u_plus_z + wz;

            // 3. Aktualizacja pozycji: x_new = x + dt * v
            x += dt * vx;
            y += dt * vy;
            z += dt * vz;
            // --- KONIEC ALGORITMU BORISA ---

            bool is_out = grid_device_vew.device_out_of_boundary_check(x, y, z);
            if(is_out){
                local_view.device_mark_for_removal(i);
            }

            // Zapis zaktualizowanych danych z powrotem do buforów SoA
            local_view.px[i] = x;
            local_view.py[i] = y;
            local_view.pz[i] = z;

            local_view.vx[i] = vx;
            local_view.vy[i] = vy;
            local_view.vz[i] = vz;
        });
    }
};