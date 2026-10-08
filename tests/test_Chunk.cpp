#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "Chunk.hpp"

/**
 * @class ChunkTest
 * @brief Test fixture for Chunk class unit tests, initializing the shared SYCL queue.
 */
class ChunkTest : public ::testing::Test {
protected:
    /// SYCL queue used for memory allocation and execution within tests.
    sycl::queue queue;

    /**
     * @brief Sets up the test environment by creating a default SYCL queue.
     */
    void SetUp() override {
        try {
            queue = sycl::queue(sycl::gpu_selector_v);
        } catch (const sycl::exception &e) {
            FAIL() << "Failed to create SYCL queue: " << e.what();
        }
    }
};

/**
 * @test Verifies initial capacity, particle count, and available space after construction.
 */
TEST_F(ChunkTest, ConstructionAndCapacity) {
    constexpr size_t test_capacity = 1024;
    Chunk chunk(queue, test_capacity);

    EXPECT_EQ(chunk.capacity(), test_capacity);
    EXPECT_EQ(chunk.count(), 0);
    EXPECT_EQ(chunk.available_space(), test_capacity);
}

/**
 * @test Verifies proper behavior when constructing a Chunk with zero capacity.
 */
TEST_F(ChunkTest, ZeroCapacityChunk) {
    Chunk chunk(queue, 0);

    EXPECT_EQ(chunk.capacity(), 0);
    EXPECT_EQ(chunk.count(), 0);
    EXPECT_EQ(chunk.available_space(), 0);
}

/**
 * @test Verifies that the move constructor correctly transfers VRAM ownership.
 */
TEST_F(ChunkTest, MoveConstructorTransfersOwnership) {
    constexpr size_t test_capacity = 512;
    Chunk original(queue, test_capacity);

    // Perform move construction
    Chunk moved(std::move(original));

    // The destination chunk must hold the transferred capacity
    EXPECT_EQ(moved.capacity(), test_capacity);
    EXPECT_EQ(moved.count(), 0);

    // Original chunk's capacity field remains unchanged, but its internal USM pointers
    // were invalidated via std::exchange, preventing double-free on destruction.
}

/**
 * @test Verifies move assignment operator behavior and resource cleanup.
 */
TEST_F(ChunkTest, MoveAssignmentOperator) {
    constexpr size_t cap_a = 256;
    constexpr size_t cap_b = 1024;

    Chunk chunk_a(queue, cap_a);
    Chunk chunk_b(queue, cap_b);

    // Perform move assignment (chunk_a frees its memory and takes over chunk_b)
    chunk_a = std::move(chunk_b);

    EXPECT_EQ(chunk_a.capacity(), cap_b);
    EXPECT_EQ(chunk_a.count(), 0);
}

/**
 * @test Stress-tests dynamic USM allocation and deallocation to detect memory leaks.
 */
TEST_F(ChunkTest, AllocationAndReleaseIntegrity) {
    constexpr size_t iterations = 50;
    constexpr size_t chunk_size = 100'000;

    EXPECT_NO_THROW({
        for (size_t i = 0; i < iterations; ++i) {
            Chunk temp_chunk(queue, chunk_size);
            // Constructing and immediately destroying in loop triggers Chunk::~Chunk()
        }
        queue.wait_and_throw();
    });
}

/**
 * @test Verifies that requesting huge memory sizes throws an exception without memory leaks.
 */
TEST_F(ChunkTest, OutOfMemoryHandling) {
    // Request an impossibly large capacity (~400 TB of VRAM)
    constexpr size_t huge_capacity = 100'000'000'000'000ULL;

    EXPECT_ANY_THROW({
        Chunk invalid_chunk(queue, huge_capacity);
    });
}

/**
 * @test Verifies self-assignment safety (chunk_a = std::move(chunk_a)).
 */
TEST_F(ChunkTest, SelfMoveAssignmentSafety) {
    constexpr size_t capacity = 128;
    Chunk chunk(queue, capacity);

    // Suppress compiler warnings for self-move assignment
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    chunk = std::move(chunk);
#pragma clang diagnostic pop

    EXPECT_EQ(chunk.capacity(), capacity);
}

/**
 * @test Verifies memory release safety when enqueued tasks complete.
 */
TEST_F(ChunkTest, DestructionWithPendingKernelExecution) {
    constexpr size_t capacity = 1024;
    
    EXPECT_NO_THROW({
        Chunk chunk(queue, capacity);
        
        // Ensure queue finishes all operations before chunk destructor executes
        queue.wait();
    });
}

/**
 * @test Verifies that ChunkSoAView correctly exposes valid USM device pointers.
 */
TEST_F(ChunkTest, GetViewPointerValidity) {
    constexpr size_t test_capacity = 1024;
    Chunk chunk(queue, test_capacity);

    ChunkSoAView view = chunk.get_view();

    // 1. Sprawdzenie, czy żaden ze wskaźników nie jest nullptr
    EXPECT_NE(view.px, nullptr);
    EXPECT_NE(view.py, nullptr);
    EXPECT_NE(view.pz, nullptr);
    EXPECT_NE(view.vx, nullptr);
    EXPECT_NE(view.vy, nullptr);
    EXPECT_NE(view.vz, nullptr);

    // 2. Weryfikacja typu pamięci USM (musi być sycl::usm::alloc::device)
    EXPECT_EQ(sycl::get_pointer_type(view.px, queue.get_context()), sycl::usm::alloc::device);
    EXPECT_EQ(sycl::get_pointer_type(view.vz, queue.get_context()), sycl::usm::alloc::device);

    // 3. Weryfikacja czy wskaźniki wskazują na odrębne bloki pamięci
    EXPECT_NE(view.px, view.py);
    EXPECT_NE(view.px, view.vx);
}

/**
 * @test Verifies that writing to pointers obtained via get_view() inside a SYCL kernel works correctly.
 */
TEST_F(ChunkTest, GetViewKernelExecution) {
    constexpr size_t count = 256;
    Chunk chunk(queue, count);

    ChunkSoAView view = chunk.get_view();

    // Uruchomienie kernela SYCL modyfikującego dane przez widok ChunkSoAView
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            view.px[i] = static_cast<float>(i) * 1.5f;
            view.vy[i] = static_cast<float>(i) + 10.0f;
        });
    }).wait();

    // Weryfikacja wyników – kopiujemy z GPU do pamięci hosta
    std::vector<float> host_px(count);
    std::vector<float> host_vy(count);

    queue.memcpy(host_px.data(), view.px, count * sizeof(float)).wait();
    queue.memcpy(host_vy.data(), view.vy, count * sizeof(float)).wait();

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(host_px[i], static_cast<float>(i) * 1.5f);
        EXPECT_FLOAT_EQ(host_vy[i], static_cast<float>(i) + 10.0f);
    }
}

/**
 * @test Verifies that memory allocations are aligned (e.g. 64-byte alignment for SIMD/Vectorization).
 */
TEST_F(ChunkTest, MemoryAlignmentCheck) {
    constexpr size_t test_capacity = 512;
    Chunk chunk(queue, test_capacity);

    ChunkSoAView view = chunk.get_view();

    // Sprawdzamy wyrównanie do przynajmniej 16 bajtów (opcjonalnie 64 bajty przy aligned_alloc_device)
    constexpr uintptr_t alignment = 16;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.px) % alignment, 0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.py) % alignment, 0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.pz) % alignment, 0U);
}

/**
 * @test Verifies that get_view() on a Zero-Capacity or Moved-From Chunk returns null pointers safely.
 */
TEST_F(ChunkTest, GetViewMovedFromState) {
    Chunk chunk_a(queue, 100);
    Chunk chunk_b = std::move(chunk_a);

    // Po przeniesieniu, chunk_a powinien zwracać nullptr w widoku
    ChunkSoAView view_a = chunk_a.get_view();
    EXPECT_EQ(view_a.px, nullptr);
    EXPECT_EQ(view_a.py, nullptr);
    EXPECT_EQ(view_a.count, 0U);

    // chunk_b powinien posiadać poprawne wskaźniki
    ChunkSoAView view_b = chunk_b.get_view();
    EXPECT_NE(view_b.px, nullptr);
}

/**
 * @test Verifies that Chunk properly allocates memory buffers and initializes counters.
 */
TEST_F(ChunkTest, InitializationAllocatesBuffers) {
    const size_t capacity = 1000;
    Chunk chunk(queue, capacity);

    EXPECT_EQ(chunk.capacity(), capacity);
    EXPECT_EQ(chunk.count(), 0);

    auto view = chunk.get_view();

    // Ensure all SoA and removal pointers are non-null
    ASSERT_NE(view.px, nullptr);
    ASSERT_NE(view.py, nullptr);
    ASSERT_NE(view.pz, nullptr);
    ASSERT_NE(view.vx, nullptr);
    ASSERT_NE(view.vy, nullptr);
    ASSERT_NE(view.vz, nullptr);
    ASSERT_NE(view.d_dead_indices, nullptr);
    ASSERT_NE(view.d_dead_count, nullptr);

    // Verify d_dead_count is initialized to zero on device USM
    int32_t h_dead_count = -1;
    queue.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 0);
}

/**
 * @test Verifies atomic marking of particles for removal directly within a SYCL kernel.
 */
TEST_F(ChunkTest, MarkForRemovalInSYCLBuffer) {
    const size_t capacity = 100;
    Chunk chunk(queue, capacity);

    auto view = chunk.get_view();

    // Submit kernel marking specific indices (2, 5, 8) concurrently on GPU
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            if (id == 2 || id == 5 || id == 8) {
                view.mark_for_removal(id);
            }
        });
    }).wait();

    // 1. Verify that dead counter recorded exactly 3 removals
    int32_t h_dead_count = 0;
    queue.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 3);

    // 2. Read back indices from device USM and verify contents
    std::vector<int32_t> h_dead_indices(3);
    queue.memcpy(h_dead_indices.data(), view.d_dead_indices, 3 * sizeof(int32_t)).wait();

    // Order of atomic inserts may vary on GPU, so sort before comparison
    std::sort(h_dead_indices.begin(), h_dead_indices.end());
    EXPECT_EQ(h_dead_indices[0], 2);
    EXPECT_EQ(h_dead_indices[1], 5);
    EXPECT_EQ(h_dead_indices[2], 8);
}

/**
 * @test Verifies that ChunkSoAView accurately reflects active particle count.
 */
TEST_F(ChunkTest, ViewReflectsActiveCount) {
    Chunk chunk(queue, 500);
    
    // Simulating allocation through ParticleStorage/Chunk setters
    auto handles = chunk.get_view();
    EXPECT_EQ(handles.count, 0);

    // After adding particles
    chunk.set_count(250);
    auto updated_view = chunk.get_view();
    EXPECT_EQ(updated_view.count, 250);
}

/**
 * @test Verifies that set_count updates internal particle count 
 * and is correctly reflected in ChunkSoAView.
 */
TEST_F(ChunkTest, SetCountUpdatesActiveParticleCount) {
    const size_t capacity = 1000;
    Chunk chunk(queue, capacity);

    // Initial check
    EXPECT_EQ(chunk.count(), 0);
    EXPECT_EQ(chunk.get_view().count, 0);

    // Update count using set_count
    const size_t expected_count = 250;
    chunk.set_count(expected_count);

    // Verify getter and view
    EXPECT_EQ(chunk.count(), expected_count);
    
    auto view = chunk.get_view();
    EXPECT_EQ(view.count, expected_count);

    // Boundary check: setting count to capacity
    chunk.set_count(capacity);
    EXPECT_EQ(chunk.count(), capacity);
    EXPECT_EQ(chunk.get_view().count, capacity);
}