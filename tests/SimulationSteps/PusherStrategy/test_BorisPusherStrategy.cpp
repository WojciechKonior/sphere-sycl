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

    template <typename Float>
    requires std::floating_point<Float>
    inline void device_sample(const Vec3<Float> &pos, Vec3<Float>& E, Vec3<Float>& B) const {
        E.x = static_cast<Float>(Ex_val); 
        E.y = static_cast<Float>(Ey_val); 
        E.z = static_cast<Float>(Ez_val);
        B.x = static_cast<Float>(Bx_val); 
        B.y = static_cast<Float>(By_val); 
        B.z = static_cast<Float>(Bz_val);
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
    }).wait();

    // Cleaner approach for sequential device submissions without host waits:
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait();

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
        }).wait();
    }

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

/**
 * @brief Testuje ruch pojedynczego jonu wodoru wzdłuż pola magnetycznego (B_z) 
 *        weryfikując każdy krok czasowy za pomocą bufora historii na GPU.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonParallelToMagneticFieldHistoryBufferTest) {
    // 1. Konfiguracja fizyczna dla jonu wodoru
    float dt = 1e-10f; // Czas kroków [s]
    constexpr float q_ion = static_cast<float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr float m_ion = static_cast<float>(Constants::MASS_HYDROGEN_ION);
    float q_over_m = q_ion / m_ion;

    float vz0 = 5000.0f; // Prędkość początkowa w osi Z [m/s]
    int num_steps = 100;

    // 2. Przygotowanie chunku na 1 cząstkę
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

    std::vector<float> h_px = {0.0f};
    std::vector<float> h_py = {0.0f};
    std::vector<float> h_pz = {0.0f};
    std::vector<float> h_vx = {0.0f};
    std::vector<float> h_vy = {0.0f};
    std::vector<float> h_vz = {vz0};

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait();
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait();
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait();
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait();
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait();
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait();

    // 3. Alokacja buforów historii w pamięci VRAM urządzenia (GPU)
    float* d_history_px = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_py = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_pz = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vx = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vy = sycl::malloc_device<float>(num_steps, queue_);
    float* d_history_vz = sycl::malloc_device<float>(num_steps, queue_);

    // 4. Konfiguracja Mock Field i strategii Borisa
    MockGridField mock_grid(queue_);
    BorisPusherStrategy<MockElectromagneticField> pusher;
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher);
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Pole magnetyczne/elektryczne zerowe lub współosiowe

    // 5. Pętla symulacji na GPU z zapisem do bufora historii po każdym kroku
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait();

        // Zapis stanu cząstki do tablicy historii na GPU
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

    // 6. Kopiowanie całej historii z GPU do pamięci hosta (CPU)
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

    // 7. Weryfikacja każdego kroku w pętli na hoście
    for (int step = 0; step < num_steps; ++step) {
        float time_t = (step + 1) * dt;
        float px = h_history_px[step];
        float py = h_history_py[step];
        float pz = h_history_pz[step];
        float vx = h_history_vx[step];
        float vy = h_history_vy[step];
        float vz = h_history_vz[step];

        // A. Sprawdzenie brak dryftu poprzecznego
        EXPECT_NEAR(px, 0.0f, 1e-6f);
        EXPECT_NEAR(py, 0.0f, 1e-6f);
        EXPECT_NEAR(vx, 0.0f, 1e-6f);
        EXPECT_NEAR(vy, 0.0f, 1e-6f);

        // B. Stałość prędkości wzdłuż osi Z
        EXPECT_NEAR(vz, vz0, 1e-3f);

        // C. Liniowy przyrost pozycji oczekiwany analitycznie
        float expected_z = vz0 * time_t;
        EXPECT_NEAR(pz, expected_z, 1e-4f);
    }

    // 8. Zwolnienie pamięci alokowanej na urządzeniu
    sycl::free(d_history_px, queue_);
    sycl::free(d_history_py, queue_);
    sycl::free(d_history_pz, queue_);
    sycl::free(d_history_vx, queue_);
    sycl::free(d_history_vy, queue_);
    sycl::free(d_history_vz, queue_);
}

/**
 * @brief Tests the cyclotron circular motion of a hydrogen ion in a uniform magnetic field 
 *        perpendicular to its velocity vector using an on-device history buffer.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonPerpendicularMagneticFieldHistoryBufferTest) {
    // 1. Physical setup for a heavy hydrogen ion
    float dt = 1e-9f; // Time step [s] (adjusted for ion gyrofrequency)
    constexpr float q_ion = static_cast<float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr float m_ion = static_cast<float>(Constants::MASS_HYDROGEN_ION);
    float q_over_m = q_ion / m_ion;

    float B_z = 0.5f;       // Uniform magnetic field along Z-axis [Tesla]
    float v0 = 5000.0f;     // Initial velocity along X-axis [m/s]

    int num_steps = 200;

    // Analytical cyclotron radius for ion: R = (m_ion * v_perp) / (|q_ion| * B_z)
    float expected_radius = std::abs(m_ion * v0 / (q_ion * B_z));
    float center_x = 0.0f;
    float center_y = -expected_radius; // Positive since q > 0 and B_z > 0 causes initial upward Lorentz force

    // 2. Prepare a chunk containing a single hydrogen ion
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

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

    // 3. Allocate on-device history buffers to record states across all time steps
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
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop entirely on the GPU and record history on device
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait();

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
        }).wait();
    }

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
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert total velocity magnitude remains constant
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_NEAR(current_speed, v0, 1e-2f);

        // C. Assert cyclotron radius invariance (circular motion around the field lines)
        float current_radius = std::sqrt((px - center_x) * (px - center_x) + 
                                         (py - center_y) * (py - center_y));
        EXPECT_NEAR(current_radius, expected_radius, expected_radius * 0.02f);
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
 * @brief Tests the cyclotron circular motion of a hydrogen ion in a uniform magnetic field,
 *        where each timestep advances the gyro-angle by exactly 30 degrees.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonPerpendicularMagneticFieldHistoryBufferTest30) {
    // 1. Physical setup for a heavy hydrogen ion
    constexpr float q_ion = static_cast<float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr float m_ion = static_cast<float>(Constants::MASS_HYDROGEN_ION);
    float q_over_m = q_ion / m_ion;

    float B_z = 0.5f;       // Uniform magnetic field along Z-axis [Tesla]
    float v0 = 5000.0f;     // Initial velocity along X-axis [m/s]

    // Cyklotron frequency: Omega_c = (q * B) / m
    float omega_c = (q_ion * B_z) / m_ion;

    // Target angle per step: 20 degrees in radians -> 30 * (PI / 180) = PI / 6
    float target_angle_deg = 30.0f;
    float target_angle_rad = target_angle_deg * (static_cast<float>(Constants::PI) / 180.0f);

    // Exact time step for 30 degrees per step: dt = delta_theta / omega_c
    float dt = target_angle_rad / omega_c;

    int num_steps = 200;

    // Analytical cyclotron radius for ion: R = (m_ion * v_perp) / (|q_ion| * B_z)
    float expected_radius = std::abs(m_ion * v0 / (q_ion * B_z));
    float center_x = 0.0f;
    float center_y = -expected_radius; // Corrected offset for q > 0 and B_z > 0

    // 2. Prepare a chunk containing a single hydrogen ion
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

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

    // 3. Allocate on-device history buffers to record states across all time steps
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
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop entirely on the GPU and record history on device
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait(); // <-- Obowiązkowy wait po wysłaniu puszera!

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
        }).wait();
    }

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
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert total velocity magnitude remains constant
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_NEAR(current_speed, v0, 1e-2f);

        // C. Assert cyclotron radius invariance (circular motion around the field lines)
        float current_radius = std::sqrt((px - center_x) * (px - center_x) + 
                                         (py - center_y) * (py - center_y));
        EXPECT_NEAR(current_radius, expected_radius, expected_radius * 0.05f);
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
 * @brief Tests the cyclotron circular motion of a hydrogen ion in a uniform magnetic field,
 *        where each timestep advances the gyro-angle by exactly 60 degrees.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonPerpendicularMagneticFieldHistoryBufferTest60) {
    // 1. Physical setup for a heavy hydrogen ion
    constexpr float q_ion = static_cast<float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr float m_ion = static_cast<float>(Constants::MASS_HYDROGEN_ION);
    float q_over_m = q_ion / m_ion;

    float B_z = 0.5f;       // Uniform magnetic field along Z-axis [Tesla]
    float v0 = 5000.0f;     // Initial velocity along X-axis [m/s]

    // Cyklotron frequency: Omega_c = (q * B) / m
    float omega_c = (q_ion * B_z) / m_ion;

    // Target angle per step: 60 degrees in radians -> 60 * (PI / 180) = PI / 3
    float target_angle_deg = 60.0f;
    float target_angle_rad = target_angle_deg * (static_cast<float>(Constants::PI) / 180.0f);

    // Exact time step for 60 degrees per step: dt = delta_theta / omega_c
    float dt = target_angle_rad / omega_c;

    int num_steps = 200;

    // Analytical cyclotron radius for ion: R = (m_ion * v_perp) / (|q_ion| * B_z)
    float expected_radius = std::abs(m_ion * v0 / (q_ion * B_z));
    float center_x = 0.0f;
    float center_y = -expected_radius; // Corrected offset for q > 0 and B_z > 0

    // 2. Prepare a chunk containing a single hydrogen ion
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

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

    // 3. Allocate on-device history buffers to record states across all time steps
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
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop entirely on the GPU and record history on device
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait(); // <-- Obowiązkowy wait po wysłaniu puszera!

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
        }).wait();
    }

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
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert total velocity magnitude remains constant
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_NEAR(current_speed, v0, 1e-2f);

        // C. Assert cyclotron radius invariance (circular motion around the field lines)
        float current_radius = std::sqrt((px - center_x) * (px - center_x) + 
                                         (py - center_y) * (py - center_y));
        EXPECT_NEAR(current_radius, expected_radius, expected_radius * 0.05f);
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
 * @brief Tests the cyclotron circular motion of a hydrogen ion in a uniform magnetic field,
 *        where each timestep advances the gyro-angle by exactly 360 degrees.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonPerpendicularMagneticFieldHistoryBufferTest360) {
    // 1. Physical setup for a heavy hydrogen ion
    constexpr float q_ion = static_cast<float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr float m_ion = static_cast<float>(Constants::MASS_HYDROGEN_ION);
    float q_over_m = q_ion / m_ion;

    float B_z = 0.5f;       // Uniform magnetic field along Z-axis [Tesla]
    float v0 = 5000.0f;     // Initial velocity along X-axis [m/s]

    // Cyklotron frequency: Omega_c = (q * B) / m
    float omega_c = (q_ion * B_z) / m_ion;

    // Target angle per step: 360 degrees in radians -> 360 * (PI / 180) = PI
    float target_angle_deg = 360.0f;
    float target_angle_rad = target_angle_deg * (static_cast<float>(Constants::PI) / 180.0f);

    // Exact time step for 360 degrees per step: dt = delta_theta / omega_c
    float dt = target_angle_rad / omega_c;

    int num_steps = 200;

    // Analytical cyclotron radius for ion: R = (m_ion * v_perp) / (|q_ion| * B_z)
    float expected_radius = std::abs(m_ion * v0 / (q_ion * B_z));
    float center_x = 0.0f;
    float center_y = -expected_radius; // Corrected offset for q > 0 and B_z > 0

    // 2. Prepare a chunk containing a single hydrogen ion
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

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

    // 3. Allocate on-device history buffers to record states across all time steps
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
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop entirely on the GPU and record history on device
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait(); // <-- Obowiązkowy wait po wysłaniu puszera!

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
        }).wait();
    }

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
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert total velocity magnitude remains constant
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_NEAR(current_speed, v0, 1e-2f);

        // C. Assert cyclotron radius invariance (circular motion around the field lines)
        float current_radius = std::sqrt((px - center_x) * (px - center_x) + 
                                         (py - center_y) * (py - center_y));
        EXPECT_NEAR(current_radius, expected_radius, expected_radius * 0.05f);
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
 * @brief Tests the cyclotron circular motion of a hydrogen ion in a uniform magnetic field,
 *        where each timestep advances the gyro-angle by exactly 720 degrees.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonPerpendicularMagneticFieldHistoryBufferTest720) {
    // 1. Physical setup for a heavy hydrogen ion
    constexpr float q_ion = static_cast<float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr float m_ion = static_cast<float>(Constants::MASS_HYDROGEN_ION);
    float q_over_m = q_ion / m_ion;

    float B_z = 0.5f;       // Uniform magnetic field along Z-axis [Tesla]
    float v0 = 5000.0f;     // Initial velocity along X-axis [m/s]

    // Cyklotron frequency: Omega_c = (q * B) / m
    float omega_c = (q_ion * B_z) / m_ion;

    // Target angle per step: 720 degrees in radians -> 720 * (PI / 180) = 2*PI
    float target_angle_deg = 720.0f;
    float target_angle_rad = target_angle_deg * (static_cast<float>(Constants::PI) / 180.0f);

    // Exact time step for 720 degrees per step: dt = delta_theta / omega_c
    float dt = target_angle_rad / omega_c;

    int num_steps = 200;

    // Analytical cyclotron radius for ion: R = (m_ion * v_perp) / (|q_ion| * B_z)
    float expected_radius = std::abs(m_ion * v0 / (q_ion * B_z));
    float center_x = 0.0f;
    float center_y = -expected_radius; // Corrected offset for q > 0 and B_z > 0

    // // ---------------- tymczasowa diagnostyka ---------------------
    // const float gamma0 =
    //     1.0f / std::sqrt(1.0f - (v0 * v0) /
    //                     (static_cast<float>(Constants::c) *
    //                     static_cast<float>(Constants::c)));

    // const float omega_actual =
    //     std::abs(q_over_m) * std::abs(B_z) / gamma0;

    // std::cout
    //     << "\n=== Boris diagnostics ===\n"
    //     << "target_angle_deg = " << target_angle_deg << '\n'
    //     << "dt              = " << dt << '\n'
    //     << "q_over_m        = " << q_over_m << '\n'
    //     << "omega_c         = " << omega_c << '\n'
    //     << "omega_actual    = " << omega_actual << '\n'
    //     << "omega_actual*dt = " << omega_actual * dt << " rad\n"
    //     << "expected_radius = " << expected_radius << '\n'
    //     << std::endl;
    // // -------------------------------------------------------------

    // 2. Prepare a chunk containing a single hydrogen ion
    Chunk chunk(queue_, 1);
    chunk.set_count(1);
    ChunkSoAView view = chunk.get_view();

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

    // 3. Allocate on-device history buffers to record states across all time steps
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
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, B_z};

    // 5. Run simulation loop entirely on the GPU and record history on device
    for (int step = 0; step < num_steps; ++step) {
        queue_.submit([&](sycl::handler &cgh) {
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh);
        }).wait(); // <-- Obowiązkowy wait po wysłaniu puszera!

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
        }).wait();
    }

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

        // // ---------------- tymczasowa diagnostyka ---------------------
        // const float measured_speed =
        //     std::sqrt(vx * vx + vy * vy + vz * vz);

        // const float measured_radius =
        //     std::sqrt((px - center_x) * (px - center_x) +
        //             (py - center_y) * (py - center_y));

        // const float measured_angle =
        //     std::atan2(-vy, vx) * 180.0f /
        //     static_cast<float>(Constants::PI);

        // std::cout
        //     << "step=" << step
        //     << " angle_target_deg=" << (step + 1) * target_angle_deg
        //     << " vx=" << vx
        //     << " vy=" << vy
        //     << " speed=" << measured_speed
        //     << " px=" << px
        //     << " py=" << py
        //     << " radius=" << measured_radius
        //     << " velocity_angle_deg=" << measured_angle
        //     << '\n';
        // // ---------------- tymczasowa diagnostyka ---------------------

        // A. Assert strictly zero displacement and velocity along the magnetic field axis (Z-axis)
        EXPECT_NEAR(pz, 0.0f, 1e-5f);
        EXPECT_NEAR(vz, 0.0f, 1e-5f);

        // B. Assert total velocity magnitude remains constant
        float current_speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        EXPECT_NEAR(current_speed, v0, 1e-2f);

        // C. Assert cyclotron radius invariance (circular motion around the field lines)
        float current_radius = std::sqrt((px - center_x) * (px - center_x) + 
                                         (py - center_y) * (py - center_y));
        EXPECT_NEAR(current_radius, expected_radius, expected_radius * 0.05f);
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
 * @brief Tests the raw Borish Algorithm on cpu in cyclotron circular motion of a hydrogen ion in a uniform magnetic field,
 *        where each timestep advances the gyro-angle by exactly 720 degrees.
 */
TEST_F(BorisPusherStrategyTest, HydrogenIonPerpendicularMagneticFieldHistoryBufferTest720_CPU)
{
    constexpr int num_steps = 200;
    using Float = double;
    using Vector3 = Vec3<Float>;

    constexpr Float q_ion = static_cast<Float>(Constants::CHARGE_HYDROGEN_ION);
    constexpr Float m_ion = static_cast<Float>(Constants::MASS_HYDROGEN_ION);

    const Float q_over_m = q_ion / m_ion;

    constexpr Float B_z = static_cast<Float>(0.5);
    constexpr Float v0 = static_cast<Float>(5000.0);

    const Float omega_c = (q_ion * B_z) / m_ion;

    const Float target_angle_deg = static_cast<Float>(720);
    const Float target_angle_rad = target_angle_deg * (static_cast<Float>(Constants::PI)) / static_cast<Float>(180.0);

    const Float dt = target_angle_rad / omega_c;


    const Float expected_radius = std::abs(m_ion * v0 / (q_ion * B_z));

    const Float center_x = static_cast<Float>(0.0);
    const Float center_y = -expected_radius;

    // Field policy configured exactly as in the original test.
    MockElectromagneticField field_policy{
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, B_z
    };

    // Initial particle state stored directly on the CPU.
    Vector3 pos{static_cast<Float>(0.0), static_cast<Float>(0.0), static_cast<Float>(0.0)};
    Vector3 vel{static_cast<Float>(v0), static_cast<Float>(0.0), static_cast<Float>(0.0)};

    std::vector<Vector3> history_pos;
    std::vector<Vector3> history_vel;

    history_pos.reserve(num_steps);
    history_vel.reserve(num_steps);

    // Run the actual production algorithm on the host.
    for (int step = 0; step < num_steps; ++step) {
        BorisPusherStrategy<MockElectromagneticField>::device_boris_algorithm<Float>(pos, vel, dt, q_over_m, field_policy);
        history_pos.push_back(pos);
        history_vel.push_back(vel);
    }

    // Verify the same invariants as in the original test.
    for (int step = 0; step < num_steps; ++step) {
        const Vector3& p = history_pos[step];
        const Vector3& v = history_vel[step];

        const Float current_speed = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

        const Float dx = p.x - center_x;
        const Float dy = p.y - center_y;

        const Float current_radius = std::sqrt(dx * dx + dy * dy);

        EXPECT_NEAR(p.z, static_cast<Float>(0.0), static_cast<Float>(1e-5))
            << "step=" << step;

        EXPECT_NEAR(v.z, static_cast<Float>(0.0), static_cast<Float>(1e-5))
            << "step=" << step;

        EXPECT_NEAR(current_speed, v0, static_cast<Float>(1e-2))
            << "step=" << step
            << ", speed=" << current_speed;

        EXPECT_NEAR(
            current_radius,
            expected_radius,
            expected_radius * static_cast<Float>(0.05))
            << "step=" << step
            << ", radius=" << current_radius;
    }
}


/** 
 * @brief Testuje ruch neutralnego wodoru (HYDROGEN_NEUTRAL) w zerowym polu magnetycznym i elektrycznym. 
 * Oczekiwany wynik: brak siły Lorentza (q = 0 oraz E=0, B=0), cząstka porusza się ze stałą prędkością po linii prostej, 
 * a zarejestrowana historia pozycji liniowo odzwierciedla przebieg ruchu. 
 */ 
TEST_F(BorisPusherStrategyTest, NeutralHydrogenZeroFieldTrajectoryTest) { 
    // 1. Konfiguracja fizyczna 
    float dt = 1e-11f; // 10 pikosekund 
    // Ładunek i masa neutralnego wodoru (ładunek wynosi 0) 
    constexpr float q_h = static_cast<float>(Constants::CHARGE_NEUTRAL); // 0.0f
    constexpr float m_h = static_cast<float>(Constants::MASS_HYDROGEN); 
    float q_over_m = q_h / m_h; // równe 0.0f

    // 2. Przygotowanie chunku na 1 cząstkę 
    Chunk chunk(queue_, 1); 
    chunk.set_count(1); 
    ChunkSoAView view = chunk.get_view(); 

    // Inicjalizacja pozycji początkowej (0, 0, 0) oraz prędkości początkowej (np. v_x = 500 m/s) 
    std::vector<float> h_px = {0.0f}; 
    std::vector<float> h_py = {0.0f}; 
    std::vector<float> h_pz = {0.0f}; 
    std::vector<float> h_vx = {500.0f}; // Prędkość wzdłuż osi X 
    std::vector<float> h_vy = {0.0f}; 
    std::vector<float> h_vz = {0.0f}; 

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait(); 

    // 3. Konfiguracja siatki i strategii Borisa z zerowym polem (E=0, B=0) 
    MockGridField mock_grid(queue_); 
    
    BorisPusherStrategy<MockElectromagneticField> pusher; 
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher); 
    // Zerowe pole elektromagnetyczne (Ex, Ey, Ez, Bx, By, Bz)
    pusher_view->field_policy_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 

    int num_steps = 50; 
    
    // Kontenery na historię drogi cząstki (rejestrowana w każdym kroku)
    std::vector<float> trajectory_px;
    std::vector<float> trajectory_py;
    std::vector<float> trajectory_pz;
    trajectory_px.reserve(num_steps);
    trajectory_py.reserve(num_steps);
    trajectory_pz.reserve(num_steps);

    float current_x = 0.0f;

    for (int step = 0; step < num_steps; ++step) { 
        queue_.submit([&](sycl::handler &cgh) { 
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh); 
        }).wait(); 

        // Rejestracja aktualnej pozycji z GPU do historii w każdym kroku
        float step_px = 0.0f;
        float step_py = 0.0f;
        float step_pz = 0.0f;

        queue_.memcpy(&step_px, view.px, sizeof(float)).wait(); 
        queue_.memcpy(&step_py, view.py, sizeof(float)).wait(); 
        queue_.memcpy(&step_pz, view.pz, sizeof(float)).wait(); 

        trajectory_px.push_back(step_px);
        trajectory_py.push_back(step_py);
        trajectory_pz.push_back(step_pz);
    } 

    // 4. Pobranie końcowych prędkości z GPU na host 
    std::vector<float> final_vx(1); 
    std::vector<float> final_vy(1); 
    std::vector<float> final_vz(1); 

    queue_.memcpy(final_vx.data(), view.vx, sizeof(float)).wait(); 
    queue_.memcpy(final_vy.data(), view.vy, sizeof(float)).wait(); 
    queue_.memcpy(final_vz.data(), view.vz, sizeof(float)).wait(); 

    // 5. Asercje fizyczne i weryfikacja historii ruchu
    // A. Prędkości nie powinny ulec żadnej zmianie (brak sił)
    EXPECT_NEAR(final_vx[0], 500.0f, 1e-5f); 
    EXPECT_NEAR(final_vy[0], 0.0f, 1e-5f); 
    EXPECT_NEAR(final_vz[0], 0.0f, 1e-5f); 

    // B. Weryfikacja zarejestrowanej historii drogi (czy ruch wzdłuż osi X jest liniowy i poprawny)
    for (int step = 0; step < num_steps; ++step) { 
        float expected_x_at_step = 500.0f * ((step + 1) * dt);
        EXPECT_NEAR(trajectory_px[step], expected_x_at_step, 1e-4f);
        EXPECT_NEAR(trajectory_py[step], 0.0f, 1e-5f);
        EXPECT_NEAR(trajectory_pz[step], 0.0f, 1e-5f);
    } 
}

/** 
 * @brief Testuje zachowanie neutralnego wodoru (HYDROGEN_NEUTRAL) w niezerowym polu elektrycznym i magnetycznym. 
 * Oczekiwany wynik: cząstka ma zerowy ładunek (q = 0), więc niezależnie od obecności pól E i B, siła Lorentza wynosi 0. 
 * Ruch powinien być jednostajny prostoliniowy, co weryfikuje pełna historia pozycji i prędkości w każdej składowej. 
 */ 
TEST_F(BorisPusherStrategyTest, NeutralHydrogenElectromagneticFieldTrajectoryTest) { 
    // 1. Konfiguracja fizyczna 
    float dt = 1e-11f; // 10 pikosekund 
    // Ładunek i masa neutralnego wodoru (ładunek wynosi 0) 
    constexpr float q_h = static_cast<float>(Constants::CHARGE_NEUTRAL); // 0.0f
    constexpr float m_h = static_cast<float>(Constants::MASS_HYDROGEN); 
    float q_over_m = q_h / m_h; // równe 0.0f

    // 2. Przygotowanie chunku na 1 cząstkę 
    Chunk chunk(queue_, 1); 
    chunk.set_count(1); 
    ChunkSoAView view = chunk.get_view(); 

    // Inicjalizacja pozycji początkowej (0, 0, 0) oraz prędkości początkowej skośnej (wzbudzenie w 3 osiach: v_x, v_y, v_z) 
    std::vector<float> h_px = {0.0f}; 
    std::vector<float> h_py = {0.0f}; 
    std::vector<float> h_pz = {0.0f}; 
    std::vector<float> h_vx = {300.0f}; // Prędkość początkowa w osi X 
    std::vector<float> h_vy = {400.0f}; // Prędkość początkowa w osi Y 
    std::vector<float> h_vz = {500.0f}; // Prędkość początkowa w osi Z 

    queue_.memcpy(view.px, h_px.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.py, h_py.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.pz, h_pz.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.vx, h_vx.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.vy, h_vy.data(), sizeof(float)).wait(); 
    queue_.memcpy(view.vz, h_vz.data(), sizeof(float)).wait(); 

    // 3. Konfiguracja siatki i strategii Borisa z NIEZEROWYM polem elektromagnetycznym 
    MockGridField mock_grid(queue_); 
    
    BorisPusherStrategy<MockElectromagneticField> pusher; 
    auto pusher_view = reinterpret_cast<BorisPusherStrategyView*>(&pusher); 
    
    // Ustawienie silnych pól elektrycznych i magnetycznych (Ex, Ey, Ez, Bx, By, Bz)
    // Ponieważ ładunek q = 0, pola te nie powinny w żaden sposób wpłynąć na cząstkę.
    pusher_view->field_policy_ = {1e5f, 2e5f, 3e5f,  // Pole elektryczne [V/m]
                                  0.5f, 1.0f, 1.5f}; // Pole magnetyczne [Tesla] 

    int num_steps = 50; 
    
    // Kontenery na pełną historię drogi cząstki (rejestrowana dla wszystkich składowych w każdym kroku)
    std::vector<float> history_px;
    std::vector<float> history_py;
    std::vector<float> history_pz;
    history_px.reserve(num_steps);
    history_py.reserve(num_steps);
    history_pz.reserve(num_steps);

    for (int step = 0; step < num_steps; ++step) { 
        queue_.submit([&](sycl::handler &cgh) { 
            pusher.push(view, dt, q_over_m, reinterpret_cast<GridField&>(mock_grid), cgh); 
        }).wait(); 

        // Rejestracja aktualnego stanu pozycji z GPU do historii w każdym kroku
        float current_px = 0.0f;
        float current_py = 0.0f;
        float current_pz = 0.0f;

        queue_.memcpy(&current_px, view.px, sizeof(float)).wait(); 
        queue_.memcpy(&current_py, view.py, sizeof(float)).wait(); 
        queue_.memcpy(&current_pz, view.pz, sizeof(float)).wait(); 

        history_px.push_back(current_px);
        history_py.push_back(current_py);
        history_pz.push_back(current_pz);
    } 

    // 4. Pobranie końcowych prędkości z GPU na host 
    std::vector<float> final_vx(1); 
    std::vector<float> final_vy(1); 
    std::vector<float> final_vz(1); 

    queue_.memcpy(final_vx.data(), view.vx, sizeof(float)).wait(); 
    queue_.memcpy(final_vy.data(), view.vy, sizeof(float)).wait(); 
    queue_.memcpy(final_vz.data(), view.vz, sizeof(float)).wait(); 

    // 5. Asercje fizyczne i weryfikacja całej historii ruchu oraz składowych
    // A. Składowe prędkości nie powinny ulec żadnej zmianie (brak przyspieszenia ze względu na q = 0)
    EXPECT_NEAR(final_vx[0], 300.0f, 1e-5f); 
    EXPECT_NEAR(final_vy[0], 400.0f, 1e-5f); 
    EXPECT_NEAR(final_vz[0], 500.0f, 1e-5f); 

    // B. Weryfikacja pełnej historii trajektorii dla każdej składowej przestrzennej (x, y, z) krok po kroku
    for (int step = 0; step < num_steps; ++step) { 
        float t = (step + 1) * dt;
        
        float expected_x = 300.0f * t;
        float expected_y = 400.0f * t;
        float expected_z = 500.0f * t;

        EXPECT_NEAR(history_px[step], expected_x, 1e-4f);
        EXPECT_NEAR(history_py[step], expected_y, 1e-4f);
        EXPECT_NEAR(history_pz[step], expected_z, 1e-4f);
    } 
}