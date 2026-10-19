#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include "SimulationSteps/ParticleInjectorStrategy/VelocityDistributionPolicy.hpp" // Dostosuj ścieżkę do pliku nagłówkowego

/**
 * @class MaxwellBoltzmannDistributionTest
 * @brief Test fixture for MaxwellBoltzmannDistribution unit tests.
 */
class MaxwellBoltzmannDistributionTest : public ::testing::Test {
protected:
    sycl::queue queue_;
    double test_temperature_;
    double test_mass_;

    MaxwellBoltzmannDistributionTest() 
        : queue_(sycl::gpu_selector_v), // Domyślny selektor (automatycznie wybierze GPU lub CPU)
          test_temperature_(300.0),         // Temperatura testowa w kelwinach [K] (np. temperatura pokojowa)
          test_mass_(3.995e-26)             // Przykładowa masa cząstki (np. atom Argonu w kg)
    {}

    void SetUp() override {
        // Inicjalizacja środowiska testowego
    }

    void TearDown() override {
        // Czyszczenie po testach
    }
};

/**
 * @brief Test batch generation of thermal velocities for 100 particles on the CPU host.
 * 
 * Verifies that the apply method correctly modifies base velocities across all spatial 
 * dimensions, yields non-zero distribution spreads, satisfies statistical expectations 
 * for individual components (mean close to zero, standard deviation consistent with v_th), 
 * and checks the total velocity magnitude statistics.
 */
TEST_F(MaxwellBoltzmannDistributionTest, BatchGeneration100Particles) {
    MaxwellBoltzmannDistribution dist(test_temperature_);
    
    const int num_particles = 100;
    std::vector<float> vx_list(num_particles, 0.0f);
    std::vector<float> vy_list(num_particles, 0.0f);
    std::vector<float> vz_list(num_particles, 0.0f);

    // Expected thermal velocity standard deviation for a single dimension: v_th = sqrt(k_B * T / m)
    float expected_v_th = static_cast<float>(std::sqrt(Constants::k_B * test_temperature_ / test_mass_));

    // Generate velocities for 100 particles using unique seed indices
    for (int i = 0; i < num_particles; ++i) {
        uint64_t seed_idx = static_cast<uint64_t>(i + 1);
        dist.apply(seed_idx, vx_list[i], vy_list[i], vz_list[i], test_mass_);
    }

    // 1. Check if all velocities were successfully modified across all components
    int modified_count = 0;
    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    float sum_vz = 0.0f;

    for (int i = 0; i < num_particles; ++i) {
        if (vx_list[i] != 0.0f && vy_list[i] != 0.0f && vz_list[i] != 0.0f) {
            modified_count++;
        }
        sum_vx += vx_list[i];
        sum_vy += vy_list[i];
        sum_vz += vz_list[i];
    }

    EXPECT_EQ(modified_count, num_particles);

    // 2. Statistical check: Mean velocities for each component should be close to 0
    float mean_vx = sum_vx / num_particles;
    float mean_vy = sum_vy / num_particles;
    float mean_vz = sum_vz / num_particles;

    EXPECT_NEAR(mean_vx, 0.0f, 0.5f * expected_v_th);
    EXPECT_NEAR(mean_vy, 0.0f, 0.5f * expected_v_th);
    EXPECT_NEAR(mean_vz, 0.0f, 0.5f * expected_v_th);

    // 3. Statistical check: Standard deviation for each component should match theoretical v_th
    float var_x = 0.0f, var_y = 0.0f, var_z = 0.0f;
    for (int i = 0; i < num_particles; ++i) {
        var_x += (vx_list[i] - mean_vx) * (vx_list[i] - mean_vx);
        var_y += (vy_list[i] - mean_vy) * (vy_list[i] - mean_vy);
        var_z += (vz_list[i] - mean_vz) * (vz_list[i] - mean_vz);
    }
    
    float measured_v_th_x = std::sqrt(var_x / num_particles);
    float measured_v_th_y = std::sqrt(var_y / num_particles);
    float measured_v_th_z = std::sqrt(var_z / num_particles);

    EXPECT_NEAR(measured_v_th_x, expected_v_th, 0.4f * expected_v_th);
    EXPECT_NEAR(measured_v_th_y, expected_v_th, 0.4f * expected_v_th);
    EXPECT_NEAR(measured_v_th_z, expected_v_th, 0.4f * expected_v_th);

    // 4. Check total velocity magnitudes (v_magnitude = sqrt(vx^2 + vy^2 + vz^2))
    float sum_v_magnitude = 0.0f;
    for (int i = 0; i < num_particles; ++i) {
        float v_mag = std::sqrt(vx_list[i] * vx_list[i] + vy_list[i] * vy_list[i] + vz_list[i] * vz_list[i]);
        sum_v_magnitude += v_mag;
        
        // Ensure individual magnitude is non-negative and physically reasonable
        EXPECT_GE(v_mag, 0.0f);
    }
    
    float mean_v_magnitude = sum_v_magnitude / num_particles;
    
    // Mean speed in 3D Maxwell-Boltzmann distribution is given by: <v> = sqrt(8 * k_B * T / (pi * m)) = sqrt(8/pi) * v_th
    // float expected_mean_speed = std::static_pointer_cast<float>(/* ... */); // (uproszczenie obliczeniowe poniżej)
    float expected_mean_speed_val = static_cast<float>(std::sqrt(8.0 / 3.14159265358979323846) * expected_v_th);

    EXPECT_NEAR(mean_v_magnitude, expected_mean_speed_val, 0.5f * expected_v_th);
}