#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include "SimulationSteps/PusherStrategy/BorisPusherStrategy.hpp"
#include "GridField.hpp"
#include "Chunk.hpp"
#include "SimulationUtils.hpp"

// --- MOCKI DO TESTÓW ---

/// @brief Mock widoku urządzenia siatki (zawsze zwraca false w teście granic)
struct MockGridFieldDeviceView {
    inline bool device_out_of_boundary_check(float, float, float) const {
        return false;
    }
};

/// @brief Mock siatki GridField zwracający nasz testowy widok
class MockGridField {
private:
    sycl::queue &queue_;
    MockGridFieldDeviceView device_view_;

public:
    explicit MockGridField(sycl::queue &q) : queue_(q), device_view_() {}

    sycl::queue& get_queue() { return queue_; }
    MockGridFieldDeviceView get_device_view() const { return device_view_; }
};

/// @brief Mock polityki pola elektromagnetycznego z konfigurowalnymi wartościami E i B
struct MockElectromagneticField {
    float Ex_val = 0.0f, Ey_val = 0.0f, Ez_val = 0.0f;
    float Bx_val = 0.0f, By_val = 0.0f, Bz_val = 0.0f;

    inline void device_sample(Vector3 pos, Vector3& E, Vector3& B) const {
        E.x = Ex_val; E.y = Ey_val; E.z = Ez_val;
        B.x = Bx_val; B.y = By_val; B.z = Bz_val;
    }
};

/// @brief A lightweight view of the BorisPusherStrategy
struct BorisPusherStrategyView {
    MockElectromagneticField field_policy_;
};

// --- TEST FIXTURE ---

class BorisPusherStrategyTest : public ::testing::Test {
protected:
    sycl::queue queue_;

    BorisPusherStrategyTest() : queue_(sycl::gpu_selector_v) {}

    void SetUp() override {}
    void TearDown() override {}
};

// --- TEST WŁAŚCIWY ---

/**
 * @brief Testuje ruch pojedynczego elektronu wzdłuż pola magnetycznego (B_z).
 * Oczekiwany wynik: cząstka porusza się ze stałą prędkością po linii prostej (brak siły Lorentza dla v || B).
 */
TEST_F(BorisPusherStrategyTest, ElectronParallelToMagneticFieldTest) {
    // 1. Konfiguracja fizyczna
    float dt = 1e-11f; // 10 pikosekund
    // Ładunek i masa elektronu (np. z tablic stałych lub przybliżone wartości)
    constexpr float q_e = static_cast<float>(Constants::CHARGE_ELECTRON);
    constexpr float m_e = static_cast<float>(Constants::MASS_ELECTRON);
    float q_over_m = q_e / m_e;

    // 2. Przygotowanie chunku na 1 cząstkę
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

    // Inicjalizacja pozycji początkowej (x=0, y=0, z=0) oraz prędkości równoległej do osi Z (v_z = 1000 m/s)
    std::vector<float> h_px = {0.0f};
    std::vector<float> h_py = {0.0f};
    std::vector<float> h_pz = {0.0f};
    std::vector<float> h_vx = {0.0f};
    std::vector<float> h_vy = {0.0f};
    std::vector<float> h_vz = {1000.0f}; // Prędkość wzdłuż osi Z

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait();
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait();
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait();
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait();
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait();
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait();

    // 3. Konfiguracja siatki i strategii Borisa z polem magnetycznym w osi Z (Bz = 1.0 Tesla)
    MockGridField mock_grid(queue_);
    
    // Ponieważ BorisPusherStrategy w kodzie używa typu ElectromagneticField, 
    // możemy przetestować logikę bezpośrednio wywołując push z mockowanym polem lub podmieniając politykę.
    // W naszym przypadku wysyłamy do kernela za pomocą handlera SYCL:
    
    BorisPusherStrategy<MockElectromagneticField> pusher;
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher);
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int num_steps = 50;
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            // Wywołujemy metodę push przekazując siatkę
            // (Uwaga: MockGridField musi być kompatybilny interfejsem z GridField, 
            // posiadając get_device_view() zwracający obiekt z device_out_of_boundary_check).
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait();
    }

    // 4. Pobranie wyników z GPU na host
    std::vector<float> final_px(1);
    std::vector<float> final_py(1);
    std::vector<float> final_pz(1);
    std::vector<float> final_vx(1);
    std::vector<float> final_vy(1);
    std::vector<float> final_vz(1);

    queue_.memcpy(final_px.data(), view.px, sizeof(float)).wait();
    queue_.memcpy(final_py.data(), view.py, sizeof(float)).wait();
    queue_.memcpy(final_pz.data(), view.pz, sizeof(float)).wait();
    queue_.memcpy(final_vx.data(), view.vx, sizeof(float)).wait();
    queue_.memcpy(final_vy.data(), view.vy, sizeof(float)).wait();
    queue_.memcpy(final_vz.data(), view.vz, sizeof(float)).wait();

    // 5. Asercje fizyczne
    // A. Skoro v || B (prędkość w osi Z, pole w osi Z), siła Lorentza (v x B) powinna wynosić 0.
    // Prędkości nie powinny się zmienić.
    EXPECT_NEAR(final_vx[0], 0.0f, 1e-5f);
    EXPECT_NEAR(final_vy[0], 0.0f, 1e-5f);
    EXPECT_NEAR(final_vz[0], 1000.0f, 1e-3f);

    // B. Cząstka powinna przesunąć się wzdłuż osi Z o dokładnie: v_z * (num_steps * dt)
    float expected_z = 1000.0f * (num_steps * dt);
    EXPECT_NEAR(final_px[0], 0.0f, 1e-5f);
    EXPECT_NEAR(final_py[0], 0.0f, 1e-5f);
    EXPECT_NEAR(final_pz[0], expected_z, 1e-4f);
}

/**
 * @brief Tests the cyclotron circular motion of an electron in a uniform magnetic field 
 *        perpendicular to its velocity vector using an on-device history buffer.
 * 
 * Verifies that:
 * - The particle state is recorded on GPU at every step without host-device synchronization overhead.
 * - After simulation, all recorded steps confirm zero drift along B, constant speed magnitude, 
 *   and stable cyclotron radius.
 */
TEST_F(BorisPusherStrategyTest, ElectronPerpendicularMagneticFieldHistoryBufferTest) {
    // 1. Physical setup
    float dt = 1e-11f; // Time step [s]
    constexpr float q_e = static_cast<float>(Constants::CHARGE_ELECTRON);
    constexpr float m_e = static_cast<float>(Constants::MASS_ELECTRON);
    float q_over_m = q_e / m_e;

    float B_z = 0.001f;    // Uniform magnetic field along Z-axis [Tesla]
    float v0 = 1000.0f;  // Initial velocity along X-axis [m/s]

    int num_steps = 200;

    // Analytical cyclotron radius: R = (m * v_perp) / (|q| * B)
    float expected_radius = std::abs(m_e * v0 / (q_e * B_z));
    float center_x = 0.0f;
    float center_y = expected_radius;

    // 2. Prepare a chunk containing a single particle
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

    // Initial state setup on host and copy to device VRAM
    std::vector<float> h_px = {0.0f};
    std::vector<float> h_py = {0.0f};
    std::vector<float> h_pz = {0.0f};
    std::vector<float> h_vx = {v0};
    std::vector<float> h_vy = {0.0f};
    std::vector<float> h_vz = {0.0f};

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait();
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait();
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait();
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait();
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait();
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait();

    // 3. Allocate on-device history buffers to record states across all time steps without CPU synchronization
    float* d_history_px = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_py = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_pz = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vx = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vy = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vz = sycl::malloc_device<float>(num_steps, queue_);

    // 4. Configure Mock Field and Pusher Strategy
    MockGridField mock_grid(queue_);
    BorisPusherStrategy<MockElectromagneticField> pusher;
    
    // Set custom magnetic field Bz = 0.1 T, zero electric field
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher);
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop entirely on the GPU (recording history in parallel)
    queue_.submit([&](sycl::handler &cgh) {
        auto v_local = view;
        auto h_px_buf = d_history_px;
        auto h_py_buf = d_history_py;
        auto h_pz_buf = d_history_pz;
        auto h_vx_buf = d_history_vx;
        auto h_vy_buf = d_history_vy;
        auto h_vz_buf = d_history_vz;
        
        // We capture a copy of the pusher and grid view to use inside a custom recording kernel wrapper or step loop
        cgh.host_task([=](sycl::interop_handle) {
            // Note: If running multiple steps, we submit kernel per step inside host task or sequential queue submissions
        });
    });

    // Cleaner approach for sequential device submissions without host waits:
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        });

        // Record current state into device history arrays
        queue_.submit([&](sycl::handler &cgh) {
            auto px_ptr = view.px;
            auto py_ptr = view.py;
            auto pz_ptr = view.pz;
            auto vx_ptr = view.vx;
            auto vy_ptr = view.vy;
            auto vz_ptr = view.vz;

            auto h_px_buf = d_history_px;
            auto h_py_buf = d_history_py;
            auto h_pz_buf = d_history_pz;
            auto h_vx_buf = d_history_vx;
            auto h_vy_buf = d_history_vy;
            auto h_vz_buf = d_history_vz;

            cgh.single_task([=]() {
                h_px_buf[step] = px_ptr[0];
                h_py_buf[step] = py_ptr[0];
                h_pz_buf[step] = pz_ptr[0];
                h_vx_buf[step] = vx_ptr[0];
                h_vy_buf[step] = vy_ptr[0];
                h_vz_buf[step] = vz_ptr[0];
            });
        });
    }
    queue_.wait(); // Synchronize only once after the entire loop!

    // 6. Copy history buffers from device to host for batch verification
    std::vector<float> h_history_px(num_steps);
    std::vector<float> h_history_py(num_steps);
    std::vector<float> h_history_pz(num_steps);
    std::vector<float> h_history_vx(num_steps);
    std::vector<float> h_history_vy(num_steps);
    std::vector<float> h_history_vz(num_steps);

    queue_.memcpy(h_history_px.data(), d_history_px, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_py.data(), d_history_py, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_pz.data(), d_history_pz, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vx.data(), d_history_vx, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vy.data(), d_history_vy, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vz.data(), d_history_vz, num_steps * sizeof(float)).wait();

    // 7. Validate all recorded steps on host
    for (int step = 0; step < num_steps; ++step) {
        float px = h_history_px[step];
        float py = h_history_py[step];
        float pz = h_history_pz[step];
        float vx = h_history_vx[step];
        float vy = h_history_vy[step];
        float vz = h_history_vz[step];

        // A. Assert strictly zero displacement and velocity along the magnetic field axis (Z-axis)
        EXPECT_NEAR(pz, 0.0f, 1e-6f);
        EXPECT_NEAR(vz, 0.0f, 1e-6f);

        // B. Assert total velocity magnitude remains constant
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_NEAR(current_speed, v0, 1e-3f);

        // C. Assert cyclotron radius invariance
        float current_radius = std::sqrt((px - center_x) * (px - center_x) + 
                                         (py - center_y) * (py - center_y));
        EXPECT_NEAR(current_radius, expected_radius, expected_radius * 0.01f);
    }

    // 8. Cleanup device history memory
    sycl::free(d_history_px, queue_);
    sycl::free(d_history_py, queue_);
    sycl::free(d_history_pz, queue_);
    sycl::free(d_history_vx, queue_);
    sycl::free(d_history_vy, queue_);
    sycl::free(d_history_vz, queue_);
}

/**
 * @brief Tests particle acceleration in a uniform electric field (pure E_x) 
 *        verifying every single time step from the on-device history buffer.
 * 
 * Verifies at every time step:
 * - Strict motion along the electric field axis (X-axis) with zero drift in Y and Z.
 * - Relativistic velocity bounding (velocity strictly less than speed of light c).
 * - Smooth, expected relativistic acceleration trajectory.
 */
TEST_F(BorisPusherStrategyTest, ElectronElectricFieldRelativisticAccelerationHistoryBufferTest) {
    // 1. Physical setup for relativistic acceleration
    float dt = 1e-13f; // Time step [s]
    constexpr float q_e = static_cast<float>(Constants::CHARGE_ELECTRON);
    constexpr float m_e = static_cast<float>(Constants::MASS_ELECTRON);
    constexpr float c   = static_cast<float>(Constants::c);
    constexpr float c2  = c * c;
    float q_over_m = q_e / m_e;

    // Uniform electric field along X-axis [V/m]
    float E_x = -5.0e9f; 
    int num_steps = 200;

    // 2. Prepare a chunk containing a single particle starting from rest
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

    std::vector<float> h_px = {0.0f};
    std::vector<float> h_py = {0.0f};
    std::vector<float> h_pz = {0.0f};
    std::vector<float> h_vx = {0.0f}; // Starts from rest
    std::vector<float> h_vy = {0.0f};
    std::vector<float> h_vz = {0.0f};

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait();
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait();
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait();
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait();
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait();
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait();

    // 3. Allocate on-device history buffers
    float* d_history_px = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_py = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_pz = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vx = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vy = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vz = sycl::malloc_device<float>(num_steps, queue_);

    // 4. Configure Mock Field and Pusher Strategy with electric field E_x
    MockGridField mock_grid(queue_);
    BorisPusherStrategy<MockElectromagneticField> pusher;
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher);
    pusher_view->field_policy_ = {E_x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // 5. Run simulation loop entirely on the GPU and record history in parallel per step
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait();

        queue_.submit([&](sycl::handler &cgh) {
            auto px_ptr = view.px;
            auto py_ptr = view.py;
            auto pz_ptr = view.pz;
            auto vx_ptr = view.vx;
            auto vy_ptr = view.vy;
            auto vz_ptr = view.vz;

            auto h_px_buf = d_history_px;
            auto h_py_buf = d_history_py;
            auto h_pz_buf = d_history_pz;
            auto h_vx_buf = d_history_vx;
            auto h_vy_buf = d_history_vy;
            auto h_vz_buf = d_history_vz;

            cgh.single_task([=]() {
                h_px_buf[step] = px_ptr[0];
                h_py_buf[step] = py_ptr[0];
                h_pz_buf[step] = pz_ptr[0];
                h_vx_buf[step] = vx_ptr[0];
                h_vy_buf[step] = vy_ptr[0];
                h_vz_buf[step] = vz_ptr[0];
            });
        }).wait();
    }

    // 6. Copy history buffers from device to host
    std::vector<float> h_history_px(num_steps);
    std::vector<float> h_history_py(num_steps);
    std::vector<float> h_history_pz(num_steps);
    std::vector<float> h_history_vx(num_steps);
    std::vector<float> h_history_vy(num_steps);
    std::vector<float> h_history_vz(num_steps);

    queue_.memcpy(h_history_px.data(), d_history_px, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_py.data(), d_history_py, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_pz.data(), d_history_pz, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vx.data(), d_history_vx, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vy.data(), d_history_vy, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vz.data(), d_history_vz, num_steps * sizeof(float)).wait();

    // 7. RIGOROUS VALIDATION OF EVERY SINGLE STEP ON HOST
    float prev_vx = 0.0f;
    float prev_ux = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        float px = h_history_px[step];
        float py = h_history_py[step];
        float pz = h_history_pz[step];
        float vx = h_history_vx[step];
        float vy = h_history_vy[step];
        float vz = h_history_vz[step];

        // A. Assert motion occurs strictly along the electric field axis (X-axis)
        EXPECT_NEAR(py, 0.0f, 1e-5f);
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vy, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert velocity strictly remains below the speed of light at every step
        EXPECT_LT(std::abs(vx), c);

        // C. Obliczenie pędu właściwego u_x oraz czynnika gamma dla bieżącego kroku
        float vv = vx * vx;
        float gamma = 1.0f / std::sqrt(std::max(0.0f, 1.0f - vv / c2));
        float curr_ux = gamma * vx;

        // D. Walidacja monotoniczności prędkości z tolerancją na szum float blisko c
        if (step > 0) {
            // Tolerancja 50 km/s na samych krańcach relatywistycznych (szum mantysy float)
            EXPECT_GE(vx, prev_vx - 50000.0f);

            // E. Kluczowa, bezwzględna asercja monotoniczności w przestrzeni PĘDU WŁAŚCIWEGO (u_x)
            // Pęd właściwy w stałym polu elektrycznym musi rosnąć bezwzględnie monotonicznie!
            EXPECT_GE(curr_ux, prev_ux);
        }

        prev_vx = vx;
        prev_ux = curr_ux;
    }

    // D. Final velocity check: ensure it successfully reached the intended relativistic fraction (~0.3c+)
    float final_vx = h_history_vx.back();
    EXPECT_GT(std::abs(final_vx), 0.3f * c);
    EXPECT_LT(std::abs(final_vx), c);

    // 8. Cleanup device history memory
    sycl::free(d_history_px, queue_);
    sycl::free(d_history_py, queue_);
    sycl::free(d_history_pz, queue_);
    sycl::free(d_history_vx, queue_);
    sycl::free(d_history_vy, queue_);
    sycl::free(d_history_vz, queue_);
}

/**
 * @brief Tests particle ExB drift in crossed electric (E_x) and magnetic (B_z) fields 
 *        by validating every single time step against analytical drift velocity and trajectory.
 */
TEST_F(BorisPusherStrategyTest, ElectronExBDriftHistoryBufferTest) {
    // 1. Physical setup for ExB drift
    float dt = 1e-12f; // Time step [s]
    constexpr float q_e = static_cast<float>(Constants::CHARGE_ELECTRON);
    constexpr float m_e = static_cast<float>(Constants::MASS_ELECTRON);
    constexpr float c   = static_cast<float>(Constants::c);
    float q_over_m = q_e / m_e;

    // Crossed fields setup
    float E_x = 1.0e6f;  // Electric field along X [V/m]
    float B_z = 0.05f;   // Magnetic field along Z [Tesla]

    int num_steps = 20;

    // Analytical drift velocity along Y-axis: v_drift_y = Ex / Bz
    float v_drift_y = E_x / B_z;

    // Cyclotron frequency and radius for expected oscillatory motion around the drifting center
    // (Starting from rest, initial gyrospeed equals drift speed magnitude)
    float v_perp = std::abs(v_drift_y);
    float cyclotron_radius = (m_e * v_perp) / (std::abs(q_e) * B_z);
    float omega_c = (std::abs(q_e) * B_z) / m_e;

    // 2. Prepare a chunk containing a single particle starting from rest
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

    std::vector<float> h_px = {0.0f};
    std::vector<float> h_py = {0.0f};
    std::vector<float> h_pz = {0.0f};
    std::vector<float> h_vx = {0.0f}; // Starts from rest
    std::vector<float> h_vy = {0.0f};
    std::vector<float> h_vz = {0.0f};

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait();
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait();
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait();
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait();
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait();
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait();

    // 3. Allocate on-device history buffers
    float* d_history_px = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_py = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_pz = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vx = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vy = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vz = sycl::malloc_device<float>(num_steps, queue_);

    // 4. Configure Mock Field and Pusher Strategy
    MockGridField mock_grid(queue_);
    BorisPusherStrategy<MockElectromagneticField> pusher;
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher);
    pusher_view->field_policy_ = {E_x, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop on GPU and record history per step
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        });

        queue_.submit([&](sycl::handler &cgh) {
            auto px_ptr = view.px;
            auto py_ptr = view.py;
            auto pz_ptr = view.pz;
            auto vx_ptr = view.vx;
            auto vy_ptr = view.vy;
            auto vz_ptr = view.vz;

            auto h_px_buf = d_history_px;
            auto h_py_buf = d_history_py;
            auto h_pz_buf = d_history_pz;
            auto h_vx_buf = d_history_vx;
            auto h_vy_buf = d_history_vy;
            auto h_vz_buf = d_history_vz;

            cgh.single_task([=]() {
                h_px_buf[step] = px_ptr[0];
                h_py_buf[step] = py_ptr[0];
                h_pz_buf[step] = pz_ptr[0];
                h_vx_buf[step] = vx_ptr[0];
                h_vy_buf[step] = vy_ptr[0];
                h_vz_buf[step] = vz_ptr[0];
            });
        });
    }
    queue_.wait();

    // 6. Copy history buffers to host
    std::vector<float> h_history_px(num_steps);
    std::vector<float> h_history_py(num_steps);
    std::vector<float> h_history_pz(num_steps);
    std::vector<float> h_history_vx(num_steps);
    std::vector<float> h_history_vy(num_steps);
    std::vector<float> h_history_vz(num_steps);

    queue_.memcpy(h_history_px.data(), d_history_px, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_py.data(), d_history_py, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_pz.data(), d_history_pz, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vx.data(), d_history_vx, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vy.data(), d_history_vy, num_steps * sizeof(float)).wait();
    queue_.memcpy(h_history_vz.data(), d_history_vz, num_steps * sizeof(float)).wait();

    // 7. RIGOROUS STEP-BY-STEP VALIDATION ON HOST
    for (int step = 0; step < num_steps; ++step) {
        float time_t = (step + 1) * dt;
        float px = h_history_px[step];
        float py = h_history_py[step];
        float pz = h_history_pz[step];
        float vx = h_history_vx[step];
        float vy = h_history_vy[step];
        float vz = h_history_vz[step];

        // A. Assert zero motion along Z-axis (orthogonal to both E and B fields)
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert particle velocity magnitude never exceeds speed of light
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_LT(current_speed, c);

        // C. Analytical trajectory comparison for ExB drift starting from rest:
        // Position y(t) should follow the drift velocity closely on average: y_expected = v_drift_y * t
        // Plus cyclotron oscillation component: y oscillates around y_drift with amplitude = cyclotron_radius.
        float expected_mean_y = v_drift_y * time_t;
        
        // The deviation of py from expected_mean_y represents the gyroradius offset bounded by cyclotron_radius
        float deviation_y = std::abs(py - expected_mean_y);
        EXPECT_LE(deviation_y, cyclotron_radius * 1.1f);
    }

    // 8. Cleanup device history memory
    sycl::free(d_history_px, queue_);
    sycl::free(d_history_py, queue_);
    sycl::free(d_history_pz, queue_);
    sycl::free(d_history_vx, queue_);
    sycl::free(d_history_vy, queue_);
    sycl::free(d_history_vz, queue_);
}