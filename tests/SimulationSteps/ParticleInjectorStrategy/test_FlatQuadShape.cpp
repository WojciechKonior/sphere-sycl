#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "SimulationSteps/ParticleInjectorStrategy/FlatQuadeShape.hpp"
#include "SimulationUtils.hpp"

/**
 * @class FlatQuadShapeTest
 * @brief Test fixture for FlatQuadShape unit and property tests.
 */
class FlatQuadShapeTest : public ::testing::Test {
protected:
    sycl::queue queue_; ///< SYCL queue for device execution.

    /// @brief Constructor initializing standard test queue.
    FlatQuadShapeTest() : queue_(sycl::gpu_selector_v) {}

    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @brief Comprehensive test for FlatQuadShape constructor and geometric properties:
 *        - Verifies successful construction and exact area calculation for valid coplanar vertices.
 *        - Verifies automatic inward normal vector derivation.
 *        - Verifies exception throwing when non-coplanar vertices are provided.
 */
TEST_F(FlatQuadShapeTest, ConstructorAndPropertiesValidation) {
    // 1. Poprawne punkty współpłaszczyznowe (kwadrat 1x1 w płaszczyźnie XY, z = 0)
    Vector3 p0{0.0f, 0.0f, 0.0f};
    Vector3 p1{1.0f, 0.0f, 0.0f};
    Vector3 p2{1.0f, 1.0f, 0.0f};
    Vector3 p3{0.0f, 1.0f, 0.0f};

    Vector3 velocity{0.0f, 0.0f, 500.0f}; // Prędkość skierowana wzdłuż osi Z
    double dt = 1e-5;                    // 10 mikrosekund

    // Sprawdzenie, czy poprawny czworobok tworzy się bezbłędnie
    EXPECT_NO_THROW({
        FlatQuadShape quad(p0, p1, p2, p3, velocity, dt);

        // A. Sprawdzenie pola powierzchni (1.0 * 1.0 = 1.0)
        EXPECT_NEAR(quad.compute_area(), 1.0, 1e-5);
        EXPECT_NEAR(quad.get_area(), 1.0, 1e-5);

        // B. Sprawdzenie automatycznie wyznaczonego wektora normalnego (powinien wskazywać wzdłuż osi Z)
        Vector3 normal = quad.device_get_inward_normal();
        EXPECT_NEAR(std::abs(normal.z), 1.0f, 1e-5);
        EXPECT_NEAR(normal.x, 0.0f, 1e-5);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5);
    });

    // 2. Weryfikacja odporności na błędy: punkty NIEWSPÓŁPŁASZCZYZNOWE (p3 odchylony w osi Z)
    Vector3 invalid_p3{0.0f, 1.0f, 0.5f};

    EXPECT_THROW({
        FlatQuadShape invalid_quad(p0, p1, p2, invalid_p3, velocity, dt);
    }, std::runtime_error);
}

/**
 * @brief Test verifying compute_area, get_area, and device_get_inward_normal 
 *        across 5 diverse coplanar quadrilateral shapes.
 */
TEST_F(FlatQuadShapeTest, MultipleShapesAreaAndNormalVerification) {
    Vector3 dummy_velocity{0.0f, 0.0f, 100.0f};
    double dt = 1e-5;

    // ==========================================
    // PRZYPADEK 1: Standardowy kwadrat 1x1 w płaszczyźnie XY (z = 0)
    // ==========================================
    {
        FlatQuadShape quad(
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            dummy_velocity, dt
        );

        EXPECT_NEAR(quad.compute_area(), 1.0, 1e-5);
        EXPECT_NEAR(quad.get_area(), 1.0, 1e-5);
        
        Vector3 normal = quad.device_get_inward_normal();
        EXPECT_NEAR(normal.x, 0.0f, 1e-5);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5);
        EXPECT_NEAR(std::abs(normal.z), 1.0f, 1e-5);
    }

    // ==========================================
    // PRZYPADEK 2: Prostokąt 2x3 w płaszczyźnie XY (z = 0)
    // ==========================================
    {
        FlatQuadShape quad(
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
            {2.0f, 3.0f, 0.0f},
            {0.0f, 3.0f, 0.0f},
            dummy_velocity, dt
        );

        EXPECT_NEAR(quad.compute_area(), 6.0, 1e-5);
        EXPECT_NEAR(quad.get_area(), 6.0, 1e-5);

        Vector3 normal = quad.device_get_inward_normal();
        EXPECT_NEAR(normal.x, 0.0f, 1e-5);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5);
        EXPECT_NEAR(std::abs(normal.z), 1.0f, 1e-5);
    }

    // ==========================================
    // PRZYPADEK 3: Kwadrat przesunięty w płaszczyźnie XZ (y = 5)
    // ==========================================
    {
        FlatQuadShape quad(
            {0.0f, 5.0f, 0.0f},
            {2.0f, 5.0f, 0.0f},
            {2.0f, 5.0f, 2.0f},
            {0.0f, 5.0f, 2.0f},
            dummy_velocity, dt
        );

        EXPECT_NEAR(quad.compute_area(), 4.0, 1e-5);
        EXPECT_NEAR(quad.get_area(), 4.0, 1e-5);

        Vector3 normal = quad.device_get_inward_normal();
        EXPECT_NEAR(normal.x, 0.0f, 1e-5);
        EXPECT_NEAR(std::abs(normal.y), 1.0f, 1e-5);
        EXPECT_NEAR(normal.z, 0.0f, 1e-5);
    }

    // ==========================================
    // PRZYPADEK 4: Równoległobok w płaszczyźnie XY
    // ==========================================
    {
        FlatQuadShape quad(
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
            {3.0f, 2.0f, 0.0f},
            {1.0f, 2.0f, 0.0f},
            dummy_velocity, dt
        );

        // Pole równoległoboku o podstawie 2 i wysokości 2 = 4.0
        EXPECT_NEAR(quad.compute_area(), 4.0, 1e-5);
        EXPECT_NEAR(quad.get_area(), 4.0, 1e-5);

        Vector3 normal = quad.device_get_inward_normal();
        EXPECT_NEAR(normal.x, 0.0f, 1e-5);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5);
        EXPECT_NEAR(std::abs(normal.z), 1.0f, 1e-5);
    }

    // ==========================================
    // PRZYPADEK 5: Czworobok na ukośnej płaszczyźnie (x + y + z = 0)
    // ==========================================
    {
        FlatQuadShape quad(
            {0.0f,  0.0f,  0.0f},
            {1.0f,  0.0f, -1.0f},
            {1.0f,  1.0f, -2.0f},
            {0.0f,  1.0f, -1.0f},
            dummy_velocity, dt
        );

        // Teoretyczne pole powierzchni dla tego kształtu wynosi sqrt(3) ~= 1.73205
        double expected_area = std::sqrt(3.0);
        EXPECT_NEAR(quad.compute_area(), expected_area, 1e-4);
        EXPECT_NEAR(quad.get_area(), expected_area, 1e-4);

        // Wektor normalny znormalizowany dla płaszczyzny x + y + z = 0 to (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        Vector3 normal = quad.device_get_inward_normal();
        float expected_component = 1.0f / std::sqrt(3.0f);
        EXPECT_NEAR(std::abs(normal.x), expected_component, 1e-4);
        EXPECT_NEAR(std::abs(normal.y), expected_component, 1e-4);
        EXPECT_NEAR(std::abs(normal.z), expected_component, 1e-4);
    }
}

/**
 * @brief Test verifying that FlatQuadShape constructor correctly throws an exception
 *        when non-coplanar vertices are provided.
 */
TEST_F(FlatQuadShapeTest, NonCoplanarVerticesThrowsException) {
    Vector3 dummy_velocity{0.0f, 0.0f, 100.0f};
    double dt = 1e-5;

    // Bazowe punkty poprawnego kwadratu w płaszczyźnie XY (z = 0)
    Vector3 p0{0.0f, 0.0f, 0.0f};
    Vector3 p1{1.0f, 0.0f, 0.0f};
    Vector3 p2{1.0f, 1.0f, 0.0f};

    // Scenariusz 1: p3 odchylony w osi Z (klasyczna piramida / złamana płaszczyzna)
    {
        Vector3 non_coplanar_p3{0.0f, 1.0f, 0.2f}; // z != 0
        
        EXPECT_THROW({
            FlatQuadShape quad(p0, p1, p2, non_coplanar_p3, dummy_velocity, dt);
        }, std::runtime_error);
    }

    // Scenariusz 2: p3 odchylony w osi X
    {
        Vector3 non_coplanar_p3{0.5f, 1.0f, 0.0f}; // x zburzył współpłaszczyznowość w innej płaszczyźnie
        
        // Zmieńmy bazę dla innego układu: np. płaszczyzna XZ (y = 0)
        Vector3 b0{0.0f, 0.0f, 0.0f};
        Vector3 b1{1.0f, 0.0f, 0.0f};
        Vector3 b2{1.0f, 0.0f, 1.0f};
        Vector3 invalid_b3{0.0f, 0.5f, 1.0f}; // y != 0, więc nie leży w płaszczyźnie Y=0

        EXPECT_THROW({
            FlatQuadShape quad(b0, b1, b2, invalid_b3, dummy_velocity, dt);
        }, std::runtime_error);
    }

    // Scenariusz 3: Całkowicie losowe punkty w przestrzeni 3D nie spełniające warunku iloczynu potrójnego
    {
        Vector3 r0{1.0f, 2.0f, 3.0f};
        Vector3 r1{4.0f, 5.0f, 6.0f};
        Vector3 r2{7.0f, 8.0f, 9.0f}; // r0, r1, r2 są współliniowe lub tworzą degenerację / złą płaszczyznę
        Vector3 r3{0.0f, 1.0f, 5.0f};

        EXPECT_THROW({
            FlatQuadShape quad(r0, r1, r2, r3, dummy_velocity, dt);
        }, std::runtime_error);
    }
}

/**
 * @brief GPU-based large-scale test verifying FlatQuadShape::device_generate_position:
 *        - Validates layer thickness bounds [0, thickness] using atomic counters.
 *        - Verifies uniform density distribution in the layer via mean depth convergence.
 *        - Verifies spatial convergence of the base projection to the quad's geometric center.
 *        - Uses SYCL reductions and atomic references to prevent data racing entirely on device.
 */
TEST_F(FlatQuadShapeTest, GpuFlatQuadPositionGenerationTest) {
    const size_t num_particles = 50000;

    // Definicja kwadratu 2x2 w płaszczyźnie XY (z = 0)
    Vector3 p0{0.0f, 0.0f, 0.0f};
    Vector3 p1{2.0f, 0.0f, 0.0f};
    Vector3 p2{2.0f, 2.0f, 0.0f};
    Vector3 p3{0.0f, 2.0f, 0.0f};

    Vector3 velocity{0.0f, 0.0f, 400.0f}; // Prędkość normalna = 400 m/s wzdłuż osi Z
    double dt = 1.25e-5;                 // dt = 12.5 us -> thickness = 400 * 12.5e-5 = 0.005 m (5 mm)
    
    FlatQuadShape quad(p0, p1, p2, p3, velocity, dt);
    double expected_thickness = 400.0 * dt;

    // 1. Alokacja pamięci USM na urządzenia dla wygenerowanych pozycji
    float* d_px = sycl::malloc_device<float>(num_particles, queue_);
    float* d_py = sycl::malloc_device<float>(num_particles, queue_);
    float* d_pz = sycl::malloc_device<float>(num_particles, queue_);

    // 2. Kernel generujący pozycje współbieżnie na GPU
    queue_.submit([&](sycl::handler& cgh) {
        auto t_quad = quad;
        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            uint64_t seed_idx = static_cast<uint64_t>(i) + 12345ULL;

            Vector3 pos = t_quad.device_generate_position(seed_idx);
            d_px[i] = pos.x;
            d_py[i] = pos.y;
            d_pz[i] = pos.z;
        });
    }).wait();

    // 3. Alokacja zasobów do weryfikacji statystycznej na GPU (redukcje i atomiki)
    float* d_sum_h = sycl::malloc_device<float>(1, queue_);
    float* d_sum_x = sycl::malloc_device<float>(1, queue_);
    float* d_sum_y = sycl::malloc_device<float>(1, queue_);
    int*   d_out_of_bounds = sycl::malloc_device<int>(1, queue_);

    queue_.memset(d_sum_h, 0, sizeof(float)).wait();
    queue_.memset(d_sum_x, 0, sizeof(float)).wait();
    queue_.memset(d_sum_y, 0, sizeof(float)).wait();
    queue_.memset(d_out_of_bounds, 0, sizeof(int)).wait();

    // Sycl redukcje dla bezpiecznych sum równoległych
    auto red_h = sycl::reduction(d_sum_h, sycl::plus<float>());
    auto red_x = sycl::reduction(d_sum_x, sycl::plus<float>());
    auto red_y = sycl::reduction(d_sum_y, sycl::plus<float>());

    Vector3 normal = quad.device_get_inward_normal();

    // 4. Kernel weryfikujący pozycje bezpośrednio na urządzeniu
    queue_.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_particles), red_h, red_x, red_y, 
            [=](sycl::id<1> idx, auto& h_red, auto& x_red, auto& y_red) {
                size_t i = idx[0];
                float px = d_px[i];
                float py = d_py[i];
                float pz = d_pz[i];

                // Obliczenie głębokości h w warstwie (rzut wektora (pos - p0) na normalną)
                float dx = px - p0.x;
                float dy = py - p0.y;
                float dz = pz - p0.z;
                float h = dx * normal.x + dy * normal.y + dz * normal.z;

                // Sprawdzenie czy cząstka mieści się w przedziale [0, thickness] z małym marginesem numerycznym
                if (h < -1e-5f || h > static_cast<float>(expected_thickness) + 1e-5f) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> atomic_oob(*d_out_of_bounds);
                    atomic_oob.fetch_add(1, sycl::memory_order::relaxed);
                }

                h_red += h;
                x_red += px;
                y_red += py;
            });
    }).wait();

    // 5. Pobranie lekkich wyników scalonych z powrotem na hosta
    float h_sum_h = 0.0f, h_sum_x = 0.0f, h_sum_y = 0.0f;
    int h_out_of_bounds = 0;

    queue_.memcpy(&h_sum_h, d_sum_h, sizeof(float));
    queue_.memcpy(&h_sum_x, d_sum_x, sizeof(float));
    queue_.memcpy(&h_sum_y, d_sum_y, sizeof(float));
    queue_.memcpy(&h_out_of_bounds, d_out_of_bounds, sizeof(int));
    queue_.wait();

    // Zwolnienie pamięci USM urządzeń
    sycl::free(d_px, queue_);
    sycl::free(d_py, queue_);
    sycl::free(d_pz, queue_);
    sycl::free(d_sum_h, queue_);
    sycl::free(d_sum_x, queue_);
    sycl::free(d_sum_y, queue_);
    sycl::free(d_out_of_bounds, queue_);

    // =========================================================================
    // 6. WERYFIKACIJA STATYSTYCZNA NA HOŚCIĘ (Asercje)
    // =========================================================================
    
    // A. Żadna cząstka nie może przekroczyć dozwolonych granic warstwy
    EXPECT_EQ(h_out_of_bounds, 0);

    // B. Średnia głębokość w warstwie jednorodnej powinna wynosić dokładnie thickness / 2
    double empirical_mean_h = static_cast<double>(h_sum_h) / num_particles;
    double theoretical_mean_h = expected_thickness / 2.0;
    EXPECT_NEAR(empirical_mean_h, theoretical_mean_h, 2e-3 * expected_thickness);

    // C. Środek geometryczny wygenerowanych punktów w płaszczyźnie XY 
    // Dla kwadratu od (0,0) do (2,2) środek wynosi (1.0, 1.0)
    double empirical_mean_x = static_cast<double>(h_sum_x) / num_particles;
    double empirical_mean_y = static_cast<double>(h_sum_y) / num_particles;
    EXPECT_NEAR(empirical_mean_x, 1.0, 0.01);
    EXPECT_NEAR(empirical_mean_y, 1.0, 0.01);
}

/**
 * @brief GPU-based variance and density distribution test across diverse quadrilateral shapes.
 * Uses only existing methods and direct public interface of FlatQuadShape.
 */
TEST_F(FlatQuadShapeTest, GpuMultiShapeVarianceAndDensityTest) {
    const size_t num_particles = 50000;

    struct TestCase {
        std::string name;
        Vector3 p0, p1, p2, p3;
        Vector3 velocity;
        double dt;
    };

    std::vector<TestCase> test_cases = {
        {
            "Symmetric Square 2x2",
            {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {0.0f, 2.0f, 0.0f},
            {0.0f, 0.0f, 300.0f}, 0.01
        },
        {
            "Rectangle 1x3",
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 3.0f, 0.0f}, {0.0f, 3.0f, 0.0f},
            {0.0f, 0.0f, 200.0f}, 0.005
        },
        {
            "Asymmetric Trapezoid",
            {0.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, {2.0f, 3.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 500.0f}, 0.004
        },
        {
            "Shifted Quad in XZ Plane",
            {0.0f, 4.0f, 0.0f}, {2.0f, 4.0f, 0.0f}, {2.0f, 4.0f, 2.0f}, {0.0f, 4.0f, 2.0f},
            {0.0f, 150.0f, 0.0f}, 0.02
        }
    };

    for (const auto& tc : test_cases) {
        SCOPED_TRACE("Testing shape: " + tc.name);

        FlatQuadShape quad(tc.p0, tc.p1, tc.p2, tc.p3, tc.velocity, tc.dt);
        
        Vector3 norm = quad.device_get_inward_normal();
        float expected_thickness = static_cast<float>(
            (tc.velocity.x * norm.x + tc.velocity.y * norm.y + tc.velocity.z * norm.z) * tc.dt
        );

        // Alokacja pamięci USM na urządzenia
        float* d_px = sycl::malloc_device<float>(num_particles, queue_);
        float* d_py = sycl::malloc_device<float>(num_particles, queue_);
        float* d_pz = sycl::malloc_device<float>(num_particles, queue_);

        // Przekazujemy cały obiekt quad do kernela (FlatQuadShape jest device-copyable)
        queue_.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                uint64_t seed_idx = static_cast<uint64_t>(i) + 9999ULL;

                Vector3 pos = quad.device_generate_position(seed_idx);
                d_px[i] = pos.x;
                d_py[i] = pos.y;
                d_pz[i] = pos.z;
            });
        }).wait();

        float* d_sum_h  = sycl::malloc_device<float>(1, queue_);
        float* d_sum_h2 = sycl::malloc_device<float>(1, queue_);
        int*   d_oob    = sycl::malloc_device<int>(1, queue_);

        queue_.memset(d_sum_h,  0, sizeof(float)).wait();
        queue_.memset(d_sum_h2, 0, sizeof(float)).wait();
        queue_.memset(d_oob,    0, sizeof(int)).wait();

        auto red_h  = sycl::reduction(d_sum_h,  sycl::plus<float>());
        auto red_h2 = sycl::reduction(d_sum_h2, sycl::plus<float>());

        Vector3 base_p0 = tc.p0;

        queue_.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_particles), red_h, red_h2,
                [=](sycl::id<1> idx, auto& h_red, auto& h2_red) {
                    size_t i = idx[0];
                    float dx = d_px[i] - base_p0.x;
                    float dy = d_py[i] - base_p0.y;
                    float dz = d_pz[i] - base_p0.z;
                    
                    float h = dx * norm.x + dy * norm.y + dz * norm.z;

                    if (h < -1e-5f || h > expected_thickness + 1e-5f) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> atomic_oob(*d_oob);
                        atomic_oob.fetch_add(1, sycl::memory_order::relaxed);
                    }

                    h_red  += h;
                    h2_red += (h * h);
                });
        }).wait();

        float h_sum_h = 0.0f, h_sum_h2 = 0.0f;
        int h_oob = 0;

        queue_.memcpy(&h_sum_h,  d_sum_h,  sizeof(float));
        queue_.memcpy(&h_sum_h2, d_sum_h2, sizeof(float));
        queue_.memcpy(&h_oob,    d_oob,    sizeof(int));
        queue_.wait();

        sycl::free(d_px, queue_);
        sycl::free(d_py, queue_);
        sycl::free(d_pz, queue_);
        sycl::free(d_sum_h,  queue_);
        sycl::free(d_sum_h2, queue_);
        sycl::free(d_oob,    queue_);

        EXPECT_EQ(h_oob, 0);

        double mean_h = static_cast<double>(h_sum_h) / num_particles;
        double mean_h2 = static_cast<double>(h_sum_h2) / num_particles;
        double empirical_variance = mean_h2 - (mean_h * mean_h);
        double theoretical_variance = (static_cast<double>(expected_thickness) * expected_thickness) / 12.0;

        EXPECT_NEAR(mean_h, expected_thickness / 2.0, 2e-3 * expected_thickness);
        EXPECT_NEAR(empirical_variance, theoretical_variance, 0.02 * theoretical_variance);
    }
}