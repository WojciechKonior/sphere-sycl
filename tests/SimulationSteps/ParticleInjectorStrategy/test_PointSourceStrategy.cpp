#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include "SimulationSteps/ParticleInjectorStrategy/PointSourceStrategy.hpp"
#include "Species.hpp"
#include "GridField.hpp"

/**
 * @class PointSourceStrategyTest
 * @brief Test fixture for PointSourceStrategy unit and device tests.
 */
class PointSourceStrategyTest : public ::testing::Test {
protected:
    sycl::queue queue_;           ///< SYCL queue for device execution.
    Vector3 test_center_;         ///< Center of the injection source [m].
    double test_mean_velocity_;   ///< Characteristic mean radial velocity [m/s].
    double test_sphere_radius_;   ///< Initial sphere radius R1 [m].
    double test_particle_mass_;   ///< Mass of the injected particle [kg].

    /// @brief Constructor initializing standard test parameters.
    PointSourceStrategyTest() 
        : queue_(sycl::gpu_selector_v),
          test_center_{0.0f, 0.0f, 0.0f},
          test_mean_velocity_(1000.0),
          test_sphere_radius_(0.05),     // 5 cm initial shell radius
          test_particle_mass_(3.995e-26) // Argon atom mass
    {}

    /**
     * @brief Set up the test environment before each test.
     */
    void SetUp() override {
        // Initialization if needed
    }

    /**
     * @brief Clean up the test environment after each test.
     */
    void TearDown() override {
        // Cleanup if needed
    }
};

/**
 * @brief Test compute_macro_particles_to_inject for both cases:
 *        1. Spherical shell (R1 > 0)
 *        2. Volume source sphere (R1 = 0)
 */
TEST_F(PointSourceStrategyTest, ComputeMacroParticlesBothCases) {
    sycl::queue q(sycl::default_selector_v);
    
    // Inicjalizacja składowych dokładnie według wzorca
    ParticleStorage storage(q);
    GridField grid(q);
    SpeciesRegistry species_reg;

    SpeciesID electron_id = SpeciesID::ELECTRON;
    double macro_weight = 100.0;
    species_reg.register_species(electron_id, "Electron", 9.109e-31, -1.602e-19, macro_weight);

    // Kontekst symulacji wymagający kolejki i storage
    double number_density = 1.0e18;
    double mean_v = 1000.0;
    double dt = 1e-5f; // 10 microseconds

    SimulationContext ctx{q, storage, grid, species_reg, dt};

    // ==========================================
    // PRZYPADEK 1: Powłoka sferyczna (R1 > 0)
    // ==========================================
    {
        double R1 = 0.05; // 5 cm
        PointSourceStrategy<MaxwellBoltzmannDistribution> shell_strategy(
            Vector3{0.0f, 0.0f, 0.0f}, 
            mean_v, 
            R1,
            MaxwellBoltzmannDistribution(300.0)
        );

        // Wzór: total = number_density * v_mean * surface_area * dt
        double surface_area = 4.0 * Constants::PI * R1 * R1;
        double expected_physical = number_density * mean_v * surface_area * static_cast<double>(ctx.dt);
        size_t expected_macro = static_cast<size_t>(std::lround(expected_physical / macro_weight));

        size_t calculated_macro = shell_strategy.compute_macro_particles_to_inject(ctx, electron_id, number_density);

        EXPECT_EQ(calculated_macro, expected_macro);
        EXPECT_GT(calculated_macro, 0ul);
    }

    // ==========================================
    // PRZYPADEK 2: Źródło objętościowe (R1 = 0)
    // ==========================================
    {
        double R1 = 0.0; // R1 = 0 -> kula objętościowa o promieniu R2 = v_mean * dt
        PointSourceStrategy<MaxwellBoltzmannDistribution> volume_strategy(
            Vector3{0.0f, 0.0f, 0.0f}, 
            mean_v, 
            R1,
            MaxwellBoltzmannDistribution(300.0)
        );

        // Wzór: R2 = v_mean * dt, volume = (4/3) * pi * R2^3, total = number_density * volume
        double R2 = mean_v * static_cast<double>(ctx.dt);
        double volume_sphere = (4.0 / 3.0) * Constants::PI * R2 * R2 * R2;
        double expected_physical = number_density * volume_sphere;
        size_t expected_macro = static_cast<size_t>(std::lround(expected_physical / macro_weight));

        size_t calculated_macro = volume_strategy.compute_macro_particles_to_inject(ctx, electron_id, number_density);

        EXPECT_EQ(calculated_macro, expected_macro);
        EXPECT_GT(calculated_macro, 0ul);
    }
}

/**
 * @brief GPU statistical test verifying mean radius against analytical expectations.
 * 
 * Launches 100,000 particles in parallel on the GPU device, computes particle positions 
 * via device_generate_position, and safely accumulates the sum of radii using on-device 
 * SYCL reductions to avoid data races. Compares empirical mean radius with theoretical 
 * expectation values for both volume source (R1 = 0) and spherical shell (R1 > 0) cases.
 */
TEST_F(PointSourceStrategyTest, GpuSpatialDistributionMeanRadiusTest) {
    const size_t num_particles = 100000;
    double mean_v = test_mean_velocity_;
    double dt = 1e-5f; // 10 microseconds

    // Allocate USM device memory for reduction sum accumulator
    float* d_sum = sycl::malloc_device<float>(1, queue_);

    // =========================================================================
    // CASE 1: Pure point source volume (R1 = 0) - Uniform sphere up to R2
    // =========================================================================
    {
        double R1 = 0.0;
        double R2 = mean_v * dt; // 0.01 m
        Vector3 center = test_center_;

        // Initialize sum counter on device to zero
        queue_.memset(d_sum, 0, sizeof(float)).wait();

        // Setup SYCL reduction for thread-safe accumulation
        auto sum_reduction = sycl::reduction(d_sum, sycl::plus<float>());

        queue_.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_particles), sum_reduction, [=](sycl::id<1> idx, auto& sum) {
                size_t i = idx[0];
                uint64_t seed_idx = static_cast<uint64_t>(i) + 1000;
                
                // Generate position on device
                Vector3 pos = PointSourceStrategy<>::device_generate_position(seed_idx, R1, R2, center);

                // Compute radial distance from center
                float dx = pos.x - static_cast<float>(center.x);
                float dy = pos.y - static_cast<float>(center.y);
                float dz = pos.z - static_cast<float>(center.z);
                float r = sycl::sqrt(dx * dx + dy * dy + dz * dz);

                // Atomically/safely accumulate into reduction variable
                sum += r;
            });
        }).wait();

        // Copy final aggregated sum back to host
        float h_sum = 0.0f;
        queue_.memcpy(&h_sum, d_sum, sizeof(float)).wait();

        // Calculate empirical vs theoretical mean radius (<r> = 3/4 * R2 for uniform sphere)
        float empirical_mean = h_sum / static_cast<float>(num_particles);
        float theoretical_mean = static_cast<float>(0.75 * R2);

        // Assert convergence within tight statistical tolerance (3-sigma margin for 100k samples)
        EXPECT_NEAR(empirical_mean, theoretical_mean, 0.0003f);
    }

    // =========================================================================
    // CASE 2: Spherical shell source (R1 > 0) - Decreasing density from R1 to R2
    // =========================================================================
    {
        double R1 = test_sphere_radius_; // 0.05 m
        double R2 = R1 + mean_v * dt;     // 0.06 m
        Vector3 center = test_center_;

        // Initialize sum counter on device to zero
        queue_.memset(d_sum, 0, sizeof(float)).wait();

        // Setup SYCL reduction
        auto sum_reduction = sycl::reduction(d_sum, sycl::plus<float>());

        queue_.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_particles), sum_reduction, [=](sycl::id<1> idx, auto& sum) {
                size_t i = idx[0];
                uint64_t seed_idx = static_cast<uint64_t>(i) + 2000;
                
                // Generate position on device
                Vector3 pos = PointSourceStrategy<>::device_generate_position(seed_idx, R1, R2, center);

                // Compute radial distance from center
                float dx = pos.x - static_cast<float>(center.x);
                float dy = pos.y - static_cast<float>(center.y);
                float dz = pos.z - static_cast<float>(center.z);
                float r = sycl::sqrt(dx * dx + dy * dy + dz * dz);

                sum += r;
            });
        }).wait();

        // Copy final aggregated sum back to host
        float h_sum = 0.0f;
        queue_.memcpy(&h_sum, d_sum, sizeof(float)).wait();

        // Calculate empirical vs theoretical mean radius (<r> = 1/3 * R2 + 2/3 * R1 for shell transform)
        float empirical_mean = h_sum / static_cast<float>(num_particles);
        float theoretical_mean = static_cast<float>((1.0 / 3.0) * R2 + (2.0 / 3.0) * R1);

        // Assert convergence within tight statistical tolerance
        EXPECT_NEAR(empirical_mean, theoretical_mean, 0.0003f);
    }

    // Clean up device USM memory
    sycl::free(d_sum, queue_);
}

/**
 * @brief GPU-based large-scale test verifying device_generate_velocity distribution and thermal spread.
 * 
 * Launches 100,000 particles in parallel on the GPU device. For each particle:
 * 1. Generates a valid spatial position using device_generate_position.
 * 2. Computes velocity via device_generate_velocity (combining radial drift and Maxwell-Boltzmann thermal noise).
 * 3. Extracts thermal velocity fluctuations by subtracting the expected spatial drift components.
 * 4. Safely accumulates sigma-interval fractions on device using atomic references to prevent data races.
 * 5. Verifies that thermal fluctuations match theoretical Maxwell-Boltzmann normal distribution probabilities.
 */
TEST_F(PointSourceStrategyTest, GpuDeviceVelocityDistributionTest) {
    const size_t num_particles = 100000;
    double mean_v = test_mean_velocity_; // 1000.0 m/s
    double temperature = 300.0;          // 300 K
    double mass = test_particle_mass_;   // Argon mass
    double dt = 1e-5f;                   // 10 microseconds
    double R1 = test_sphere_radius_;     // 0.05 m
    double R2 = R1 + mean_v * dt;        // 0.06 m
    Vector3 center = test_center_;

    MaxwellBoltzmannDistribution dist_policy(temperature);

    // Expected thermal velocity standard deviation per component: v_th = sqrt(k_B * T / m)
    float expected_v_th = static_cast<float>(std::sqrt(Constants::k_B * temperature / mass));

    // Allocate USM device memory for sigma interval counters (1sigma, 2sigma, 3sigma)
    int* d_count_1 = sycl::malloc_device<int>(1, queue_);
    int* d_count_2 = sycl::malloc_device<int>(1, queue_);
    int* d_count_3 = sycl::malloc_device<int>(1, queue_);

    // Initialize counters to zero on device
    queue_.memset(d_count_1, 0, sizeof(int)).wait();
    queue_.memset(d_count_2, 0, sizeof(int)).wait();
    queue_.memset(d_count_3, 0, sizeof(int)).wait();

    // Submit parallel kernel executing on GPU
    queue_.submit([&](sycl::handler& cgh) {
        auto t_dist = dist_policy;

        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            uint64_t seed_pos = static_cast<uint64_t>(i) + 10000;
            uint64_t seed_vel = static_cast<uint64_t>(i) + 20000;

            // 1. Generate spatial position on device
            Vector3 pos = PointSourceStrategy<>::device_generate_position(seed_pos, R1, R2, center);

            // 2. Generate velocity vector combining drift and thermal noise
            Vector3 vel = PointSourceStrategy<>::device_generate_velocity(
                seed_vel, mean_v, mass, pos, center, t_dist
            );

            // 3. Compute expected base drift velocity components for this position
            float dx = pos.x - static_cast<float>(center.x);
            float dy = pos.y - static_cast<float>(center.y);
            float dz = pos.z - static_cast<float>(center.z);
            float r = sycl::sqrt(dx * dx + dy * dy + dz * dz);
            float inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;

            float vx_base = dx * inv_r * static_cast<float>(mean_v);
            float vy_base = dy * inv_r * static_cast<float>(mean_v);
            float vz_base = dz * inv_r * static_cast<float>(mean_v);

            // 4. Isolate thermal velocity fluctuations (deviation from base radial drift)
            float dvx = vel.x - vx_base;
            float abs_dvx = sycl::fabs(dvx);

            // 5. Construct atomic references locally inside the kernel using raw USM pointers
            sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                                 sycl::memory_scope::device, 
                                 sycl::access::address_space::global_space> atomic_1s(*d_count_1);
            
            sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                                 sycl::memory_scope::device, 
                                 sycl::access::address_space::global_space> atomic_2s(*d_count_2);
            
            sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                                 sycl::memory_scope::device, 
                                 sycl::access::address_space::global_space> atomic_3s(*d_count_3);

            // Safely increment counters based on thermal standard deviation thresholds
            if (abs_dvx <= 1.0f * expected_v_th) {
                atomic_1s.fetch_add(1, sycl::memory_order::relaxed);
            }
            if (abs_dvx <= 2.0f * expected_v_th) {
                atomic_2s.fetch_add(1, sycl::memory_order::relaxed);
            }
            if (abs_dvx <= 3.0f * expected_v_th) {
                atomic_3s.fetch_add(1, sycl::memory_order::relaxed);
            }
        });
    }).wait();

    // Copy results back to host
    int h_count_1 = 0, h_count_2 = 0, h_count_3 = 0;
    queue_.memcpy(&h_count_1, d_count_1, sizeof(int));
    queue_.memcpy(&h_count_2, d_count_2, sizeof(int));
    queue_.memcpy(&h_count_3, d_count_3, sizeof(int));
    queue_.wait();

    // Clean up device USM memory
    sycl::free(d_count_1, queue_);
    sycl::free(d_count_2, queue_);
    sycl::free(d_count_3, queue_);

    // Compute empirical fractions of thermal fluctuations
    float frac_1sigma = static_cast<float>(h_count_1) / num_particles;
    float frac_2sigma = static_cast<float>(h_count_2) / num_particles;
    float frac_3sigma = static_cast<float>(h_count_3) / num_particles;

    // Assertions against standard normal distribution probabilities for thermal noise
    EXPECT_NEAR(frac_1sigma, 0.6827f, 0.015f); // ~68.27% within 1 sigma
    EXPECT_NEAR(frac_2sigma, 0.9545f, 0.010f); // ~95.45% within 2 sigma
    EXPECT_NEAR(frac_3sigma, 0.9973f, 0.005f); // ~99.73% within 3 sigma
}

/**
 * @brief Super ultra end-to-end GPU test for device_initialize_particles with on-device verification.
 * 
 * Verifies the complete particle injection and statistical validation pipeline entirely on the GPU:
 * 1. Allocates USM buffers and runs device_initialize_particles in parallel.
 * 2. Launches a second verification kernel on GPU to check spatial bounds, mean radius, 
 *    and Maxwell-Boltzmann thermal velocity distributions using device atomic counters to prevent data races.
 * 3. Copies only tiny scalar results back to host for final assertions.
 */
TEST_F(PointSourceStrategyTest, GpuEndToEndParticleInitializationOnDeviceTest) {
    const size_t num_particles = 50000;
    double mean_v = test_mean_velocity_; // 1000.0 m/s
    double temperature = 300.0;          // 300 K
    double mass = test_particle_mass_;   // Argon mass
    double dt = 1e-5f;                   // 10 microseconds
    double R1 = test_sphere_radius_;     // 0.05 m
    double R2 = R1 + mean_v * dt;        // 0.06 m
    Vector3 center = test_center_;

    PointSourceStrategy<MaxwellBoltzmannDistribution> strategy(
        center, mean_v, R1, MaxwellBoltzmannDistribution(temperature)
    );

    // 1. Allocate USM device memory for position and velocity components
    float* d_px = sycl::malloc_device<float>(num_particles, queue_);
    float* d_py = sycl::malloc_device<float>(num_particles, queue_);
    float* d_pz = sycl::malloc_device<float>(num_particles, queue_);
    float* d_vx = sycl::malloc_device<float>(num_particles, queue_);
    float* d_vy = sycl::malloc_device<float>(num_particles, queue_);
    float* d_vz = sycl::malloc_device<float>(num_particles, queue_);

    SegmentHandle handle{d_px, d_py, d_pz, d_vx, d_vy, d_vz, 0 /* offset */, num_particles};
    InjectionParameters params{dt, mass, 0};

    // 2. Execute particle initialization kernel on GPU
    queue_.submit([&](sycl::handler& cgh) {
        strategy.device_initialize_particles(handle, params, cgh);
    }).wait();

    // 3. Allocate USM device memory for validation accumulators and counters
    float* d_sum_r = sycl::malloc_device<float>(1, queue_);
    int* d_out_of_bounds = sycl::malloc_device<int>(1, queue_);
    int* d_count_1s = sycl::malloc_device<int>(1, queue_);
    int* d_count_2s = sycl::malloc_device<int>(1, queue_);
    int* d_count_3s = sycl::malloc_device<int>(1, queue_);

    queue_.memset(d_sum_r, 0, sizeof(float)).wait();
    queue_.memset(d_out_of_bounds, 0, sizeof(int)).wait();
    queue_.memset(d_count_1s, 0, sizeof(int)).wait();
    queue_.memset(d_count_2s, 0, sizeof(int)).wait();
    queue_.memset(d_count_3s, 0, sizeof(int)).wait();

    // Setup reduction for safe parallel summation of radii on GPU
    auto sum_reduction = sycl::reduction(d_sum_r, sycl::plus<float>());

    float expected_v_th = static_cast<float>(std::sqrt(Constants::k_B * temperature / mass));

    // 4. Submit verification kernel on GPU (Fully race-safe using reductions and atomics)
    queue_.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_particles), sum_reduction, [=](sycl::id<1> idx, auto& sum_r_red) {
            size_t i = idx[0];

            float px = d_px[i];
            float py = d_py[i];
            float pz = d_pz[i];
            float vx = d_vx[i];
            float vy = d_vy[i];
            float vz = d_vz[i];

            // --- Spatial checks ---
            float dx = px - static_cast<float>(center.x);
            float dy = py - static_cast<float>(center.y);
            float dz = pz - static_cast<float>(center.z);
            float r = sycl::sqrt(dx * dx + dy * dy + dz * dz);

            if (r < static_cast<float>(R1) - 1e-4f || r > static_cast<float>(R2) + 1e-4f) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_oob(*d_out_of_bounds);
                atomic_oob.fetch_add(1, sycl::memory_order::relaxed);
            }

            sum_r_red += r;

            // --- Velocity thermal checks ---
            float inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
            float vx_base = dx * inv_r * static_cast<float>(mean_v);
            float vy_base = dy * inv_r * static_cast<float>(mean_v);
            float vz_base = dz * inv_r * static_cast<float>(mean_v);

            float dvx = vx - vx_base;
            float abs_dvx = sycl::fabs(dvx);

            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_1s(*d_count_1s);
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_2s(*d_count_2s);
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_3s(*d_count_3s);

            if (abs_dvx <= 1.0f * expected_v_th) {
                atomic_1s.fetch_add(1, sycl::memory_order::relaxed);
            }
            if (abs_dvx <= 2.0f * expected_v_th) {
                atomic_2s.fetch_add(1, sycl::memory_order::relaxed);
            }
            if (abs_dvx <= 3.0f * expected_v_th) {
                atomic_3s.fetch_add(1, sycl::memory_order::relaxed);
            }
        });
    }).wait();

    // 5. Copy only lightweight scalar results back to host
    float h_sum_r = 0.0f;
    int h_out_of_bounds = 0;
    int h_c1 = 0, h_c2 = 0, h_c3 = 0;

    queue_.memcpy(&h_sum_r, d_sum_r, sizeof(float));
    queue_.memcpy(&h_out_of_bounds, d_out_of_bounds, sizeof(int));
    queue_.memcpy(&h_c1, d_count_1s, sizeof(int));
    queue_.memcpy(&h_c2, d_count_2s, sizeof(int));
    queue_.memcpy(&h_c3, d_count_3s, sizeof(int));
    queue_.wait();

    // Clean up all device USM memory
    sycl::free(d_px, queue_);
    sycl::free(d_py, queue_);
    sycl::free(d_pz, queue_);
    sycl::free(d_vx, queue_);
    sycl::free(d_vy, queue_);
    sycl::free(d_vz, queue_);
    sycl::free(d_sum_r, queue_);
    sycl::free(d_out_of_bounds, queue_);
    sycl::free(d_count_1s, queue_);
    sycl::free(d_count_2s, queue_);
    sycl::free(d_count_3s, queue_);

    // =========================================================================
    // 6. HOST-SIDE FINAL ASSERTIONS ON COMPACT METRICS
    // =========================================================================
    // All particles must be inside bounds
    EXPECT_EQ(h_out_of_bounds, 0);

    // Verify mean radius convergence
    double empirical_mean_r = static_cast<double>(h_sum_r) / num_particles;
    double theoretical_mean_r = (1.0 / 3.0) * R2 + (2.0 / 3.0) * R1;
    EXPECT_NEAR(empirical_mean_r, theoretical_mean_r, 0.0005);

    // Verify Maxwell-Boltzmann velocity sigma fractions
    float frac_1sigma = static_cast<float>(h_c1) / num_particles;
    float frac_2sigma = static_cast<float>(h_c2) / num_particles;
    float frac_3sigma = static_cast<float>(h_c3) / num_particles;

    EXPECT_NEAR(frac_1sigma, 0.6827f, 0.015f);
    EXPECT_NEAR(frac_2sigma, 0.9545f, 0.010f);
    EXPECT_NEAR(frac_3sigma, 0.9973f, 0.005f);
}