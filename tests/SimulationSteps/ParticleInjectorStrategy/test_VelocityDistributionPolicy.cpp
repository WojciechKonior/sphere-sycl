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
    float expected_mean_speed_val = static_cast<float>(std::sqrt(8.0 / 3.14159265358979323846) * expected_v_th);

    EXPECT_NEAR(mean_v_magnitude, expected_mean_speed_val, 0.5f * expected_v_th);
}

/**
 * @brief Large-scale GPU kernel test verifying Maxwell-Boltzmann distribution using on-device atomic reduction.
 * 
 * Launches 100,000 particles in parallel on the GPU device, applies thermal velocity offsets, 
 * and evaluates the CDF intervals directly on the device using atomic counters instantiated 
 * locally inside the kernel to prevent data races and device-copyability compilation errors.
 */
TEST_F(MaxwellBoltzmannDistributionTest, GpuLargeScaleDistributionTestOnDevice) {
    MaxwellBoltzmannDistribution dist(test_temperature_);
    
    const int num_particles = 100000;

    // Allocate device memory for counters (1sigma, 2sigma, 3sigma) using USM
    int* d_count_1 = sycl::malloc_device<int>(1, queue_);
    int* d_count_2 = sycl::malloc_device<int>(1, queue_);
    int* d_count_3 = sycl::malloc_device<int>(1, queue_);

    // Initialize counters to zero on device
    queue_.memset(d_count_1, 0, sizeof(int)).wait();
    queue_.memset(d_count_2, 0, sizeof(int)).wait();
    queue_.memset(d_count_3, 0, sizeof(int)).wait();

    double mass = test_mass_;
    float expected_v_th = static_cast<float>(std::sqrt(Constants::k_B * test_temperature_ / test_mass_));

    // Submit parallel kernel executing distribution and atomic reduction on GPU
    queue_.submit([&](sycl::handler& cgh) {
        auto t_dist = dist;
        
        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            int i = idx[0];
            uint64_t seed_idx = static_cast<uint64_t>(i) + 50000; // Unique seed per particle

            float vx = 0.0f;
            float vy = 0.0f;
            float vz = 0.0f;

            // Apply Maxwell-Boltzmann thermal velocity
            t_dist.apply(seed_idx, vx, vy, vz, mass);

            // Check absolute deviation for x-component
            float val = sycl::fabs(vx);

            // Construct atomic_ref locally inside the kernel using raw device pointers (fully device-copyable)
            sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                               sycl::memory_scope::device, 
                               sycl::access::address_space::global_space> atomic_1s(*d_count_1);
            
            sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                               sycl::memory_scope::device, 
                               sycl::access::address_space::global_space> atomic_2s(*d_count_2);
            
            sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                               sycl::memory_scope::device, 
                               sycl::access::address_space::global_space> atomic_3s(*d_count_3);

            // Safely increment device counters using atomics to prevent data races
            if (val <= 1.0f * expected_v_th) {
                atomic_1s.fetch_add(1, sycl::memory_order::relaxed);
            }
            if (val <= 2.0f * expected_v_th) {
                atomic_2s.fetch_add(1, sycl::memory_order::relaxed);
            }
            if (val <= 3.0f * expected_v_th) {
                atomic_3s.fetch_add(1, sycl::memory_order::relaxed);
            }
        });
    }).wait();

    // Copy only the final scalar results back to host
    int h_count_1 = 0, h_count_2 = 0, h_count_3 = 0;
    queue_.memcpy(&h_count_1, d_count_1, sizeof(int));
    queue_.memcpy(&h_count_2, d_count_2, sizeof(int));
    queue_.memcpy(&h_count_3, d_count_3, sizeof(int));
    queue_.wait();

    // Clean up device memory
    sycl::free(d_count_1, queue_);
    sycl::free(d_count_2, queue_);
    sycl::free(d_count_3, queue_);

    // Compute empirical fractions
    float frac_1sigma = static_cast<float>(h_count_1) / num_particles;
    float frac_2sigma = static_cast<float>(h_count_2) / num_particles;
    float frac_3sigma = static_cast<float>(h_count_3) / num_particles;

    // Assertions against theoretical standard normal distribution probabilities
    EXPECT_NEAR(frac_1sigma, 0.6827f, 0.012f); // Expected ~68.27%
    EXPECT_NEAR(frac_2sigma, 0.9545f, 0.008f); // Expected ~95.45%
    EXPECT_NEAR(frac_3sigma, 0.9973f, 0.003f); // Expected ~99.73%
}

/**
 * @brief Test guard clauses for invalid or non-physical parameters (zero/negative temperature or mass).
 * 
 * Verifies that when temperature or particle mass is less than or equal to zero, 
 * the apply method returns early without modifying the input base velocities.
 */
TEST_F(MaxwellBoltzmannDistributionTest, ExtremeInvalidParametersGuard) {
    float vx_base = 10.0f;
    float vy_base = 20.0f;
    float vz_base = 30.0f;
    uint64_t seed_idx = 42;

    // Case 1: Zero temperature
    {
        MaxwellBoltzmannDistribution zero_temp_dist(0.0);
        float vx = vx_base, vy = vy_base, vz = vz_base;
        zero_temp_dist.apply(seed_idx, vx, vy, vz, test_mass_);
        EXPECT_EQ(vx, vx_base);
        EXPECT_EQ(vy, vy_base);
        EXPECT_EQ(vz, vz_base);
    }

    // Case 2: Negative temperature
    {
        MaxwellBoltzmannDistribution neg_temp_dist(-100.0);
        float vx = vx_base, vy = vy_base, vz = vz_base;
        neg_temp_dist.apply(seed_idx, vx, vy, vz, test_mass_);
        EXPECT_EQ(vx, vx_base);
        EXPECT_EQ(vy, vy_base);
        EXPECT_EQ(vz, vz_base);
    }

    // Case 3: Negative particle mass
    {
        MaxwellBoltzmannDistribution valid_dist(300.0);
        float vx = vx_base, vy = vy_base, vz = vz_base;
        valid_dist.apply(seed_idx, vx, vy, vz, -1.0e-26); // Invalid negative mass
        EXPECT_EQ(vx, vx_base);
        EXPECT_EQ(vy, vy_base);
        EXPECT_EQ(vz, vz_base);
    }
}

/**
 * @brief Stress test with extreme high temperature (Thermonuclear plasma conditions, e.g., 1e9 K).
 * 
 * Ensures that the algorithm handles massive thermal velocities without producing 
 * NaN (Not a Number) or Inf (Infinity) values due to floating-point overflows.
 */
TEST_F(MaxwellBoltzmannDistributionTest, ExtremeHighTemperatureStress) {
    // 1 Billion Kelvin (stellar core / fusion reactor conditions)
    double extreme_temp = 1.0e9; 
    MaxwellBoltzmannDistribution extreme_dist(extreme_temp);

    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    uint64_t seed_idx = 999;

    extreme_dist.apply(seed_idx, vx, vy, vz, test_mass_);

    // Check that values are finite and valid numbers
    EXPECT_FALSE(std::isnan(vx));
    EXPECT_FALSE(std::isnan(vy));
    EXPECT_FALSE(std::isnan(vz));

    EXPECT_FALSE(std::isinf(vx));
    EXPECT_FALSE(std::isinf(vy));
    EXPECT_FALSE(std::isinf(vz));

    // Velocities should be exceptionally large
    EXPECT_GT(std::abs(vx), 1000.0f);
}

/**
 * @brief Stress test with extreme particle masses (Electron mass vs Macroparticle/Dust mass).
 * 
 * Checks numerical stability when handling extreme mass scales in the denominator.
 */
TEST_F(MaxwellBoltzmannDistributionTest, ExtremeParticleMassStress) {
    MaxwellBoltzmannDistribution dist(300.0); // Room temperature

    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    uint64_t seed_idx = 123;

    // Subtest A: Extremely light particle (Electron mass ~9.1e-31 kg)
    {
        double electron_mass = 9.109e-31;
        float vx_e = vx, vy_e = vy, vz_e = vz;
        dist.apply(seed_idx, vx_e, vy_e, vz_e, electron_mass);
        
        EXPECT_FALSE(std::isnan(vx_e));
        EXPECT_FALSE(std::isinf(vx_e));
    }

    // Subtest B: Extremely heavy particle / Dust grain (e.g., 1.0 kg macroscopic particle)
    {
        double heavy_mass = 1.0;
        float vx_h = vx, vy_h = vy, vz_h = vz;
        dist.apply(seed_idx, vx_h, vy_h, vz_h, heavy_mass);
        
        EXPECT_FALSE(std::isnan(vx_h));
        EXPECT_FALSE(std::isinf(vx_h));
    }
}

/**
 * @brief Stress test with extreme low temperature (Near absolute zero, e.g., 1e-9 K).
 * 
 * Verifies that thermal velocity spreading becomes extremely small (proportional to v_th) 
 * and doesn't introduce unwanted large numerical artifacts or NaNs, using a dynamic 
 * 5-sigma safety margin to prevent flaky test failures.
 */
TEST_F(MaxwellBoltzmannDistributionTest, ExtremeLowTemperatureStress) {
    double near_zero_temp = 1.0e-9; 
    MaxwellBoltzmannDistribution cold_dist(near_zero_temp);

    float vx_base = 50.0f;
    float vy_base = 50.0f;
    float vz_base = 50.0f;
    float vx = vx_base, vy = vy_base, vz = vz_base;
    uint64_t seed_idx = 777;

    cold_dist.apply(seed_idx, vx, vy, vz, test_mass_);

    // Values must remain finite and valid numbers
    EXPECT_FALSE(std::isnan(vx));
    EXPECT_FALSE(std::isnan(vy));
    EXPECT_FALSE(std::isnan(vz));

    // Calculate theoretical thermal velocity standard deviation dynamically
    float expected_v_th = static_cast<float>(std::sqrt(Constants::k_B * near_zero_temp / test_mass_));

    // Allow up to 5 standard deviations (5-sigma safety margin) to ensure robustness against random fluctuations
    float safety_tolerance = 5.0f * expected_v_th;

    EXPECT_NEAR(vx, vx_base, safety_tolerance);
    EXPECT_NEAR(vy, vy_base, safety_tolerance);
    EXPECT_NEAR(vz, vz_base, safety_tolerance);
}