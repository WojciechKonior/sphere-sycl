#pragma once

#include <sycl/sycl.hpp>
#include "SimulationUtils.hpp"
#include "Species.hpp"

/**
 * @brief Maxwell-Boltzmann thermal velocity distribution policy.
 * 
 * Applies thermal velocity noise (Gaussian distribution) using the Box-Muller 
 * transform driven by deterministic PCG random number generator salts. 
 * Suitable for thermal plasmas and gases in thermodynamic equilibrium.
 */
struct MaxwellBoltzmannDistribution {
    double temperature; ///< Temperature of the species in Kelvin [K].

    /// @brief Constructs a Maxwell-Boltzmann velocity distribution.
    explicit MaxwellBoltzmannDistribution(double temp) : temperature(temp) {}

    /// @brief Applies thermal velocity offsets to the base velocities on the GPU device.
    void apply(uint64_t seed_idx, float &vx_base, float &vy_base, float &vz_base, double particle_mass) const {
        if (temperature <= 0.0 || particle_mass <= 0.0) {
            return;
        }

        // Thermal velocity standard deviation for a single spatial dimension: v_th = sqrt(k_B * T / m)
        float v_th = static_cast<float>(sycl::sqrt(Constants::k_B * temperature / particle_mass));

        // Fetch independent uniform random numbers in [0, 1) using dedicated PCG salts (4 to 7)
        float u1 = sycl::fmax(SimulationUtils::pcg_random_float(seed_idx, 4), 1e-7f);
        float u2 = SimulationUtils::pcg_random_float(seed_idx, 5);
        float u3 = sycl::fmax(SimulationUtils::pcg_random_float(seed_idx, 6), 1e-7f);
        float u4 = SimulationUtils::pcg_random_float(seed_idx, 7);

        // Box-Muller transform for X and Y components
        float mag = sycl::sqrt(-2.0f * sycl::log(u1));
        float z0 = mag * sycl::cos(2.0f * static_cast<float>(Constants::PI) * u2);
        float z1 = mag * sycl::sin(2.0f * static_cast<float>(Constants::PI) * u3);

        // Box-Muller transform for Z component (using independent logarithmic magnitude)
        float mag_z = sycl::sqrt(-2.0f * sycl::log(u3));
        float z2 = mag_z * sycl::cos(2.0f * static_cast<float>(Constants::PI) * u4);

        // Add thermal spreading components to the base stream velocities
        vx_base += z0 * v_th;
        vy_base += z1 * v_th;
        vz_base += z2 * v_th;
    }
};