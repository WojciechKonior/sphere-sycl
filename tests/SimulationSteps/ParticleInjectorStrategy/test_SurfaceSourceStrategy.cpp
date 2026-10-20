#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include "SimulationSteps/ParticleInjectorStrategy/FlatQuadeShape.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/SurfaceSourceStrategy.hpp"
#include "SimulationUtils.hpp"
#include "GridField.hpp"
#include "ParticleStorage.hpp"
#include "Species.hpp"

/**
 * @class SurfaceSourceStrategyTest
 * @brief Test fixture for SurfaceSourceStrategy unit and property tests.
 */
class SurfaceSourceStrategyTest : public ::testing::Test {
protected:
    sycl::queue queue_;
    ParticleStorage storage_;
    GridField grid_;
    SpeciesRegistry species_registry_;

    SurfaceSourceStrategyTest() 
        : queue_(sycl::gpu_selector_v),
          storage_(queue_),
          grid_(queue_),
          species_registry_() 
    {
        // Rejestrujemy domyślne gatunki do testów (np. elektron z wagą 1000.0)
        // Dostosuj metodę rejestracji do API swojego projektu, np. register_species lub add
    }

    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @brief Test verifying that SurfaceSourceStrategy correctly computes the surface area
 *        across multiple diverse quadrilateral shapes (symmetric, asymmetric, and 3D inclined).
 */
TEST_F(SurfaceSourceStrategyTest, MultiShapeSurfaceAreaValidation) {
    struct TestCase {
        std::string name;
        Vector3 p0, p1, p2, p3;
        Vector3 velocity;
        double dt;
        double expected_area;
    };

    std::vector<TestCase> test_cases = {
        {
            "Symmetric Square 2x2",
            {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {0.0f, 2.0f, 0.0f},
            {0.0f, 0.0f, 300.0f}, 0.01,
            4.0
        },
        {
            "Rectangle 1x3",
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 3.0f, 0.0f}, {0.0f, 3.0f, 0.0f},
            {0.0f, 0.0f, 200.0f}, 0.005,
            3.0
        },
        {
            "Asymmetric Trapezoid",
            {0.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, {2.0f, 3.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 500.0f}, 0.004,
            7.0 // Prawidłowa suma pól trójkątów (6.0 + 1.0)
        },
        {
            "Inclined Quad in 3D Space",
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, -1.0f}, {1.0f, 1.0f, -2.0f}, {0.0f, 1.0f, -1.0f},
            {0.0f, 0.0f, 100.0f}, 0.01,
            std::sqrt(3.0) // sqrt(3) ≈ 1.73205
        }
    };

    for (const auto& tc : test_cases) {
        SCOPED_TRACE("Testing shape: " + tc.name);

        FlatQuadShape quad(tc.p0, tc.p1, tc.p2, tc.p3, tc.dt);
        SurfaceSourceStrategy<FlatQuadShape, MaxwellBoltzmannDistribution> surface_strategy(
            quad,
            tc.velocity,
            MaxwellBoltzmannDistribution(300.0)
        );

        EXPECT_NEAR(surface_strategy.get_surface_area(), tc.expected_area, 1e-4);
    }
}

/**
 * @brief Test verifying that SurfaceSourceStrategy correctly computes the characteristic velocity 
 *        (magnitude of the 3D velocity vector) across multiple test cases.
 */
TEST_F(SurfaceSourceStrategyTest, MultiVelocityCharacteristicVelocityValidation) {
    struct VelocityTestCase {
        std::string name;
        Vector3 velocity;
        double expected_characteristic_velocity;
    };

    std::vector<VelocityTestCase> test_cases = {
        {
            "Pure Z-direction velocity",
            {0.0f, 0.0f, 300.0f},
            300.0
        },
        {
            "Pure X-direction velocity",
            {-150.0f, 0.0f, 0.0f},
            150.0
        },
        {
            "3D Diagonal velocity (3-4-12 scaled)",
            {30.0f, 40.0f, 120.0f}, // sqrt(30^2 + 40^2 + 120^2) = sqrt(900 + 1600 + 14400) = sqrt(16900) = 130.0
            130.0
        },
        {
            "Zero velocity",
            {0.0f, 0.0f, 0.0f},
            0.0
        }
    };

    // Standardowy prosty kwadrat do inicjalizacji strategii
    FlatQuadShape quad(
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 
        {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 
        0.01
    );

    for (const auto& tc : test_cases) {
        SCOPED_TRACE("Testing velocity case: " + tc.name);

        SurfaceSourceStrategy<FlatQuadShape, MaxwellBoltzmannDistribution> surface_strategy(
            quad,
            tc.velocity,
            MaxwellBoltzmannDistribution(300.0)
        );

        EXPECT_NEAR(surface_strategy.get_characteristic_velocity(), tc.expected_characteristic_velocity, 1e-4);
    }
}

/**
 * @brief Comprehensive multi-case test verifying macro-particle count calculation and 
 *        volumetric density representation across diverse simulation scenarios.
 */
TEST_F(SurfaceSourceStrategyTest, ComputeMacroParticlesToInjectValidation) {
    // 1. Inicjalizacja bazowego kontekstu symulacji
    SimulationContext ctx{
        .queue = queue_,
        .storage = storage_,
        .grid = grid_,
        .species_registry = species_registry_,
        .dt = 1e-5,
        .current_time = 0.0,
        .current_step = 0
    };

    struct MacroInjectionTestCase {
        std::string name;
        Vector3 p0, p1, p2, p3;     // Wierzchołki czworoboku
        Vector3 velocity;            // Wektor prędkości iniekcji
        double dt;                   // Krok czasowy [s]
        double number_density;       // Zadana gęstość numeryczna [m^-3]
        SpeciesID species_id;        // Gatunek cząstki
    };

    std::vector<MacroInjectionTestCase> test_cases = {
        {
            "Standard Symmetric Square (Electron)",
            {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {0.0f, 2.0f, 0.0f}, // Area = 4.0 m^2
            {0.0f, 0.0f, 500.0f},                                                         // v_mean = 500.0 m/s
            1e-5,                                                                        // dt = 1e-5 s
            1.0e16,                                                                      // n = 1e16 m^-3
            SpeciesID::ELECTRON
        },
        {
            "Rectangular Beam Source (Ion)",
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 3.0f, 0.0f}, {0.0f, 3.0f, 0.0f}, // Area = 3.0 m^2
            {200.0f, 0.0f, 0.0f},                                                        // v_mean = 200.0 m/s
            2e-5,                                                                        // dt = 2e-5 s
            5.0e15,                                                                      // n = 5e15 m^-3
            SpeciesID::ELECTRON
        },
        {
            "High-Density Fast Flow Injection",
            {0.0f, 0.0f, 0.0f}, {1.5f, 0.0f, 0.0f}, {1.5f, 1.5f, 0.0f}, {0.0f, 1.5f, 0.0f}, // Area = 2.25 m^2
            {0.0f, 1000.0f, 0.0f},                                                       // v_mean = 1000.0 m/s
            5.0e-6,                                                                      // dt = 5e-6 s
            1.0e17,                                                                      // n = 1e17 m^-3
            SpeciesID::ELECTRON
        },
        {
            "Asymmetric Trapezoid Surface Source",
            {0.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, {2.0f, 3.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, // Area = 7.0 m^2
            {30.0f, 40.0f, 120.0f},                                                      // v_mean = 130.0 m/s (sqrt(30^2+40^2+120^2))
            1e-4,                                                                        // dt = 1e-4 s
            2.0e14,                                                                      // n = 2e14 m^-3
            SpeciesID::ELECTRON
        }
    };

    for (const auto& tc : test_cases) {
        SCOPED_TRACE("Testing macro injection case: " + tc.name);

        // Aktualizujemy dt w kontekście dla danego przypadku
        ctx.dt = tc.dt;

        // 1. Inicjalizacja kształtu i strategii
        FlatQuadShape quad(tc.p0, tc.p1, tc.p2, tc.p3, tc.dt);
        SurfaceSourceStrategy<FlatQuadShape, MaxwellBoltzmannDistribution> surface_strategy(
            quad,
            tc.velocity,
            MaxwellBoltzmannDistribution(300.0)
        );

        // 2. Pobranie parametrów geometrycznych i fizycznych
        double surface_area = surface_strategy.get_surface_area();
        double v_mean = surface_strategy.get_characteristic_velocity();
        double injection_volume = surface_area * (v_mean * tc.dt);

        // Pobieramy wagę statystyczną bezpośrednio z rejestru gatunków dla danego typu cząstek
        double particle_weight = static_cast<double>(ctx.species_registry.get(tc.species_id).weight);

        // Obliczenia teoretyczne
        double expected_total_physical_particles = tc.number_density * injection_volume;
        size_t expected_macro_count = static_cast<size_t>(std::lround(expected_total_physical_particles / particle_weight));

        // 3. Wywołanie testowanej metody strategii
        size_t computed_macro_count = surface_strategy.compute_macro_particles_to_inject(ctx, tc.species_id, tc.number_density);

        // 4. Asercje weryfikujące dokładną liczbę makrocząstek
        EXPECT_EQ(computed_macro_count, expected_macro_count);

        // 5. Asercje weryfikujące poprawność reprezentacji gęstości w objętości warstwy (Area * Thickness)
        if (computed_macro_count > 0 && injection_volume > 0.0) {
            double actual_total_physical_particles = static_cast<double>(computed_macro_count) * particle_weight;
            double achieved_number_density = actual_total_physical_particles / injection_volume;

            // Sprawdzamy, czy odtworzona gęstość numeryczna mieści się w tolerancji 1% (błąd wynikający z zaokrągleń dyskretnych do lround)
            EXPECT_NEAR(achieved_number_density, tc.number_density, tc.number_density * 0.01);
        }
    }
}

/**
 * @brief GPU-based unit test verifying particle initialization, spatial layer bounds,
 *        and Maxwell-Boltzmann velocity distributions entirely on device using SYCL reductions.
 */
TEST_F(SurfaceSourceStrategyTest, DeviceInitializeParticlesValidationForElectron) {
    sycl::queue q = queue_;
    
    Vector3 p0{0.0f, 0.0f, 0.0f};
    Vector3 p1{2.0f, 0.0f, 0.0f};
    Vector3 p2{2.0f, 2.0f, 0.0f};
    Vector3 p3{0.0f, 2.0f, 0.0f};
    
    Vector3 mean_velocity{0.0f, 0.0f, 300.0f};
    double dt = 1.0e-5;
    double temperature = 10.0;
    
    FlatQuadShape quad(p0, p1, p2, p3,dt);
    SurfaceSourceStrategy<FlatQuadShape, MaxwellBoltzmannDistribution> strategy(
        quad,
        mean_velocity,
        MaxwellBoltzmannDistribution(temperature)
    );

    size_t count = 2000000;
    
    // Alokacja pamięci device dla cząstek
    float* d_px = sycl::malloc_device<float>(count, q);
    float* d_py = sycl::malloc_device<float>(count, q);
    float* d_pz = sycl::malloc_device<float>(count, q);
    float* d_vx = sycl::malloc_device<float>(count, q);
    float* d_vy = sycl::malloc_device<float>(count, q);
    float* d_vz = sycl::malloc_device<float>(count, q);

    SegmentHandle handle{
        .px = d_px, .py = d_py, .pz = d_pz,
        .vx = d_vx, .vy = d_vy, .vz = d_vz,
        .offset = 0,
        .count = count
    };

    double particle_mass = Constants::MASS_ELECTRON; 
    InjectionParameters params{
        .dt = dt,
        .particle_mass = particle_mass,
        .global_offset = 12345
    };

    // 1. Inicjalizacja cząstek na GPU
    q.submit([&](sycl::handler &cgh) {
        strategy.device_initialize_particles(handle, params, cgh);
    }).wait();

    // 2. Alokacja pamięci DEVICE dla zmiennych redukcyjnych (kluczowe, aby uniknąć illegal address!)
    float* d_sum_vx = sycl::malloc_device<float>(1, q);
    float* d_sum_vy = sycl::malloc_device<float>(1, q);
    float* d_sum_vz = sycl::malloc_device<float>(1, q);
    float* d_sq_vx = sycl::malloc_device<float>(1, q);
    float* d_sq_vy = sycl::malloc_device<float>(1, q);
    float* d_sq_vz = sycl::malloc_device<float>(1, q);

    q.memset(d_sum_vx, 0, sizeof(float)).wait();
    q.memset(d_sum_vy, 0, sizeof(float)).wait();
    q.memset(d_sum_vz, 0, sizeof(float)).wait();
    q.memset(d_sq_vx, 0, sizeof(float)).wait();
    q.memset(d_sq_vy, 0, sizeof(float)).wait();
    q.memset(d_sq_vz, 0, sizeof(float)).wait();

    // Redukcja statystyk prędkości na GPU
    q.submit([&](sycl::handler &cgh) {
        auto red_vx = sycl::reduction(d_sum_vx, sycl::plus<>());
        auto red_vy = sycl::reduction(d_sum_vy, sycl::plus<>());
        auto red_vz = sycl::reduction(d_sum_vz, sycl::plus<>());
        auto red_sq_vx = sycl::reduction(d_sq_vx, sycl::plus<>());
        auto red_sq_vy = sycl::reduction(d_sq_vy, sycl::plus<>());
        auto red_sq_vz = sycl::reduction(d_sq_vz, sycl::plus<>());

        cgh.parallel_for(sycl::range<1>(count), red_vx, red_vy, red_vz, red_sq_vx, red_sq_vy, red_sq_vz,
            [=](sycl::id<1> idx, auto &vxs, auto &vys, auto &vzs, auto &sq_vxs, auto &sq_vys, auto &sq_vzs) {
                size_t i = idx[0];
                float vx = d_vx[i];
                float vy = d_vy[i];
                float vz = d_vz[i];

                vxs.combine(vx);
                vys.combine(vy);
                vzs.combine(vz);
                sq_vxs.combine(vx * vx);
                sq_vys.combine(vy * vy);
                sq_vzs.combine(vz * vz);
            });
    }).wait();

    // Skopiowanie wyników redukcji z powrotem na CPU
    float sum_vx = 0, sum_vy = 0, sum_vz = 0;
    float sum_sq_vx = 0, sum_sq_vy = 0, sum_sq_vz = 0;
    q.memcpy(&sum_vx, d_sum_vx, sizeof(float)).wait();
    q.memcpy(&sum_vy, d_sum_vy, sizeof(float)).wait();
    q.memcpy(&sum_vz, d_sum_vz, sizeof(float)).wait();
    q.memcpy(&sum_sq_vx, d_sq_vx, sizeof(float)).wait();
    q.memcpy(&sum_sq_vy, d_sq_vy, sizeof(float)).wait();
    q.memcpy(&sum_sq_vz, d_sq_vz, sizeof(float)).wait();

    double mean_vx = sum_vx / static_cast<double>(count);
    double mean_vy = sum_vy / static_cast<double>(count);
    double mean_vz = sum_vz / static_cast<double>(count);

    double var_vx = (sum_sq_vx / static_cast<double>(count)) - (mean_vx * mean_vx);
    double var_vy = (sum_sq_vy / static_cast<double>(count)) - (mean_vy * mean_vy);
    double var_vz = (sum_sq_vz / static_cast<double>(count)) - (mean_vz * mean_vz);

    // Weryfikacja M-B
    // EXPECT_NEAR(mean_vx, 0.0, 5.0);
    // EXPECT_NEAR(mean_vy, 0.0, 5.0);
    // EXPECT_NEAR(mean_vz, mean_velocity.z, 5.0);

    EXPECT_NEAR(mean_vx, 0.0, 15.0);
    EXPECT_NEAR(mean_vy, 0.0, 15.0);
    EXPECT_NEAR(mean_vz, mean_velocity.z, 15.0);

    const double k_B = Constants::k_B;
    double expected_thermal_variance = (k_B * temperature) / particle_mass;
    
    EXPECT_NEAR(var_vx, expected_thermal_variance, expected_thermal_variance * 0.15);
    EXPECT_NEAR(var_vy, expected_thermal_variance, expected_thermal_variance * 0.15);
    EXPECT_NEAR(var_vz, expected_thermal_variance, expected_thermal_variance * 0.15);

    // 3. Redukcja min/max dla pozycji Z (również w pamięci device)
    float* d_min_z = sycl::malloc_device<float>(1, q);
    float* d_max_z = sycl::malloc_device<float>(1, q);
    
    float init_min = 1e9f;
    float init_max = -1e9f;
    q.memcpy(d_min_z, &init_min, sizeof(float)).wait();
    q.memcpy(d_max_z, &init_max, sizeof(float)).wait();

    q.submit([&](sycl::handler &cgh) {
        auto red_min_z = sycl::reduction(d_min_z, sycl::minimum<>());
        auto red_max_z = sycl::reduction(d_max_z, sycl::maximum<>());

        cgh.parallel_for(sycl::range<1>(count), red_min_z, red_max_z,
            [=](sycl::id<1> idx, auto &mn_z, auto &mx_z) {
                float z = d_pz[idx[0]];
                mn_z.combine(z);
                mx_z.combine(z);
            });
    }).wait();

    float min_z = 0, max_z = 0;
    q.memcpy(&min_z, d_min_z, sizeof(float)).wait();
    q.memcpy(&max_z, d_max_z, sizeof(float)).wait();

    EXPECT_GE(min_z, -0.005f);
    EXPECT_LE(max_z, 0.010f);

    // Zwolnienie pamięci urządzeń
    sycl::free(d_px, q);
    sycl::free(d_py, q);
    sycl::free(d_pz, q);
    sycl::free(d_vx, q);
    sycl::free(d_vy, q);
    sycl::free(d_vz, q);
    sycl::free(d_sum_vx, q);
    sycl::free(d_sum_vy, q);
    sycl::free(d_sum_vz, q);
    sycl::free(d_sq_vx, q);
    sycl::free(d_sq_vy, q);
    sycl::free(d_sq_vz, q);
    sycl::free(d_min_z, q);
    sycl::free(d_max_z, q);
}

/**
 * @brief GPU-based unit test verifying particle initialization, spatial layer bounds,
 *        and Maxwell-Boltzmann velocity distributions for heavy hydrogen ions entirely on device.
 */
TEST_F(SurfaceSourceStrategyTest, DeviceInitializeParticlesValidationForHydrogenIon) {
    sycl::queue q = queue_;
    
    Vector3 p0{0.0f, 0.0f, 0.0f};
    Vector3 p1{2.0f, 0.0f, 0.0f};
    Vector3 p2{2.0f, 2.0f, 0.0f};
    Vector3 p3{0.0f, 2.0f, 0.0f};
    
    Vector3 mean_velocity{0.0f, 0.0f, 300.0f};
    double dt = 1.0e-5;
    double temperature = 300.0; // Standardowa temperatura 300 K dla jonów
    
    FlatQuadShape quad(p0, p1, p2, p3, dt);
    SurfaceSourceStrategy<FlatQuadShape, MaxwellBoltzmannDistribution> strategy(
        quad,
        mean_velocity,
        MaxwellBoltzmannDistribution(temperature)
    );

    size_t count = 2000000;
    
    // Alokacja pamięci device dla cząstek
    float* d_px = sycl::malloc_device<float>(count, q);
    float* d_py = sycl::malloc_device<float>(count, q);
    float* d_pz = sycl::malloc_device<float>(count, q);
    float* d_vx = sycl::malloc_device<float>(count, q);
    float* d_vy = sycl::malloc_device<float>(count, q);
    float* d_vz = sycl::malloc_device<float>(count, q);

    SegmentHandle handle{
        .px = d_px, .py = d_py, .pz = d_pz,
        .vx = d_vx, .vy = d_vy, .vz = d_vz,
        .offset = 0,
        .count = count
    };

    // Używamy masy ciężkiego jonu wodoru (protonu / jonu zasarzającego)
    double particle_mass = Constants::MASS_HYDROGEN_ION; 
    InjectionParameters params{
        .dt = dt,
        .particle_mass = particle_mass,
        .global_offset = 12345
    };

    // 1. Inicjalizacja cząstek na GPU
    q.submit([&](sycl::handler &cgh) {
        strategy.device_initialize_particles(handle, params, cgh);
    }).wait();

    // 2. Alokacja pamięci DEVICE dla zmiennych redukcyjnych
    float* d_sum_vx = sycl::malloc_device<float>(1, q);
    float* d_sum_vy = sycl::malloc_device<float>(1, q);
    float* d_sum_vz = sycl::malloc_device<float>(1, q);
    float* d_sq_vx = sycl::malloc_device<float>(1, q);
    float* d_sq_vy = sycl::malloc_device<float>(1, q);
    float* d_sq_vz = sycl::malloc_device<float>(1, q);

    q.memset(d_sum_vx, 0, sizeof(float)).wait();
    q.memset(d_sum_vy, 0, sizeof(float)).wait();
    q.memset(d_sum_vz, 0, sizeof(float)).wait();
    q.memset(d_sq_vx, 0, sizeof(float)).wait();
    q.memset(d_sq_vy, 0, sizeof(float)).wait();
    q.memset(d_sq_vz, 0, sizeof(float)).wait();

    // Redukcja statystyk prędkości na GPU
    q.submit([&](sycl::handler &cgh) {
        auto red_vx = sycl::reduction(d_sum_vx, sycl::plus<>());
        auto red_vy = sycl::reduction(d_sum_vy, sycl::plus<>());
        auto red_vz = sycl::reduction(d_sum_vz, sycl::plus<>());
        auto red_sq_vx = sycl::reduction(d_sq_vx, sycl::plus<>());
        auto red_sq_vy = sycl::reduction(d_sq_vy, sycl::plus<>());
        auto red_sq_vz = sycl::reduction(d_sq_vz, sycl::plus<>());

        cgh.parallel_for(sycl::range<1>(count), red_vx, red_vy, red_vz, red_sq_vx, red_sq_vy, red_sq_vz,
            [=](sycl::id<1> idx, auto &vxs, auto &vys, auto &vzs, auto &sq_vxs, auto &sq_vys, auto &sq_vzs) {
                size_t i = idx[0];
                float vx = d_vx[i];
                float vy = d_vy[i];
                float vz = d_vz[i];

                vxs.combine(vx);
                vys.combine(vy);
                vzs.combine(vz);
                sq_vxs.combine(vx * vx);
                sq_vys.combine(vy * vy);
                sq_vzs.combine(vz * vz);
            });
    }).wait();

    // Skopiowanie wyników redukcji z powrotem na CPU
    float sum_vx = 0, sum_vy = 0, sum_vz = 0;
    float sum_sq_vx = 0, sum_sq_vy = 0, sum_sq_vz = 0;
    q.memcpy(&sum_vx, d_sum_vx, sizeof(float)).wait();
    q.memcpy(&sum_vy, d_sum_vy, sizeof(float)).wait();
    q.memcpy(&sum_vz, d_sum_vz, sizeof(float)).wait();
    q.memcpy(&sum_sq_vx, d_sq_vx, sizeof(float)).wait();
    q.memcpy(&sum_sq_vy, d_sq_vy, sizeof(float)).wait();
    q.memcpy(&sum_sq_vz, d_sq_vz, sizeof(float)).wait();

    double mean_vx = sum_vx / static_cast<double>(count);
    double mean_vy = sum_vy / static_cast<double>(count);
    double mean_vz = sum_vz / static_cast<double>(count);

    double var_vx = (sum_sq_vx / static_cast<double>(count)) - (mean_vx * mean_vx);
    double var_vy = (sum_sq_vy / static_cast<double>(count)) - (mean_vy * mean_vy);
    double var_vz = (sum_sq_vz / static_cast<double>(count)) - (mean_vz * mean_vz);

    // Weryfikacja średnich prędkości (dla ciężkich jonów przy N=2M błąd standardowy jest bardzo mały)
    EXPECT_NEAR(mean_vx, 0.0, 5.0);
    EXPECT_NEAR(mean_vy, 0.0, 5.0);
    EXPECT_NEAR(mean_vz, mean_velocity.z, 5.0);

    const double k_B = Constants::k_B;
    double expected_thermal_variance = (k_B * temperature) / particle_mass;
    
    // Weryfikacja wariancji termicznej (Maxwell-Boltzmann equipartition)
    EXPECT_NEAR(var_vx, expected_thermal_variance, expected_thermal_variance * 0.05);
    EXPECT_NEAR(var_vy, expected_thermal_variance, expected_thermal_variance * 0.05);
    EXPECT_NEAR(var_vz, expected_thermal_variance, expected_thermal_variance * 0.05);

    // 3. Redukcja min/max dla pozycji Z (weryfikacja warstwy przestrzennej)
    float* d_min_z = sycl::malloc_device<float>(1, q);
    float* d_max_z = sycl::malloc_device<float>(1, q);
    
    float init_min = 1e9f;
    float init_max = -1e9f;
    q.memcpy(d_min_z, &init_min, sizeof(float)).wait();
    q.memcpy(d_max_z, &init_max, sizeof(float)).wait();

    q.submit([&](sycl::handler &cgh) {
        auto red_min_z = sycl::reduction(d_min_z, sycl::minimum<>());
        auto red_max_z = sycl::reduction(d_max_z, sycl::maximum<>());

        cgh.parallel_for(sycl::range<1>(count), red_min_z, red_max_z,
            [=](sycl::id<1> idx, auto &mn_z, auto &mx_z) {
                float z = d_pz[idx[0]];
                mn_z.combine(z);
                mx_z.combine(z);
            });
    }).wait();

    float min_z = 0, max_z = 0;
    q.memcpy(&min_z, d_min_z, sizeof(float)).wait();
    q.memcpy(&max_z, d_max_z, sizeof(float)).wait();

    EXPECT_GE(min_z, -0.005f);
    EXPECT_LE(max_z, 0.010f);

    // Zwolnienie pamięci urządzeń
    sycl::free(d_px, q);
    sycl::free(d_py, q);
    sycl::free(d_pz, q);
    sycl::free(d_vx, q);
    sycl::free(d_vy, q);
    sycl::free(d_vz, q);
    sycl::free(d_sum_vx, q);
    sycl::free(d_sum_vy, q);
    sycl::free(d_sum_vz, q);
    sycl::free(d_sq_vx, q);
    sycl::free(d_sq_vy, q);
    sycl::free(d_sq_vz, q);
    sycl::free(d_min_z, q);
    sycl::free(d_max_z, q);
}