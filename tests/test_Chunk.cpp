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

    /**
     * @brief Cleans up the test environment after each test execution.
     */
    void TearDown() override {
        try {
            queue.wait();
        } catch (const sycl::exception &e) {
            ADD_FAILURE() << "SYCL exception during TearDown wait: " << e.what();
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
    chunk.set_count(10);

    auto view = chunk.get_view();

    view.mark_for_removal(queue, 2);
    view.mark_for_removal(queue, 5);
    view.mark_for_removal(queue, 8);

    queue.wait();

    // 1. Verify that dead counter recorded exactly 3 removals
    int32_t h_dead_count = 0;
    queue.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 3);

    // 2. Read back indices from device USM and verify contents
    size_t num_word_bits = (capacity + 31) / 32;
    std::vector<uint32_t> h_is_dead(num_word_bits, 0);
    queue.memcpy(h_is_dead.data(), view.d_is_dead, num_word_bits * sizeof(uint32_t)).wait();

    // 3. Verify bits for particles 2, 5, and 8 are set, and others are 0
    auto is_bit_set = [&](size_t idx) {
        size_t word_idx = idx / 32;
        uint32_t bit_mask = 1u << (idx % 32);
        return (h_is_dead[word_idx] & bit_mask) != 0;
    };

    EXPECT_TRUE(is_bit_set(2));
    EXPECT_TRUE(is_bit_set(5));
    EXPECT_TRUE(is_bit_set(8));

    // Upewnijmy się, że inne cząstki w zakresie 0-9 (poza 2, 5, 8) nie zostały zaznaczone
    for (int i = 0; i < 10; ++i) {
        if (i == 2 || i == 5 || i == 8) continue;
        EXPECT_FALSE(is_bit_set(i));
    }
}

/**
 * @test Verifies that attempting to mark an out-of-bounds or non-existent particle 
 *       is safely rejected without modifying d_dead_count or causing device errors.
 */
TEST_F(ChunkTest, MarkNonExistentParticleSafety) {
    const size_t capacity = 100;
    Chunk chunk(queue, capacity);

    // 1. Sztucznie ustawiamy 5 aktywnych cząstek (prawidłowe indeksy: 0, 1, 2, 3, 4)
    chunk.set_count(5);

    auto view = chunk.get_view();

    // Upewniamy się, że licznik martwych cząstek jest wyzerowany na GPU
    queue.memset(view.d_dead_count, 0, sizeof(int32_t)).wait();

    // 2. Próbujemy oznaczyć prawidłową cząstkę (indeks 2) oraz NIEISTNIEJĄCE (indeksy 5, 10, 999)
    view.mark_for_removal(queue, 2);   // VALID (2 < 5)
    view.mark_for_removal(queue, 5);   // INVALID (5 >= 5 - pierwsza nieistniejąca cząstka)
    view.mark_for_removal(queue, 10);  // INVALID (10 >= 5)
    view.mark_for_removal(queue, 999); // INVALID (OOB poza capacity)

    queue.wait();

    // 3. Weryfikacja: Tylko 1 prawidłowa cząstka powinna zostać zarejestrowana
    int32_t h_dead_count = 0;
    queue.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 1);

    // 4. Odczytujemy bitmaskę z bufora na GPU i sprawdzamy, czy zaznaczony jest tylko indeks 2
    size_t num_word_bits = (capacity + 31) / 32;
    std::vector<uint32_t> h_is_dead(num_word_bits, 0);
    queue.memcpy(h_is_dead.data(), view.d_is_dead, num_word_bits * sizeof(uint32_t)).wait();

    auto is_bit_set = [&](size_t idx) {
        size_t word_idx = idx / 32;
        uint32_t bit_mask = 1u << (idx % 32);
        return (h_is_dead[word_idx] & bit_mask) != 0;
    };

    // Sprawdzamy, czy indeks 2 jest zaznaczony, a próby oznaczania pozostałych (5, 10, 999) zostały bezpiecznie odrzucone
    EXPECT_TRUE(is_bit_set(2));
    EXPECT_FALSE(is_bit_set(5));
    EXPECT_FALSE(is_bit_set(10));
    // Dla indeksu 999 (poza capacity) sprawdzamy bezpiecznie, że nie wyszliśmy poza zakres wektora hosta
    if (999 < capacity) {
        EXPECT_FALSE(is_bit_set(999));
    }
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

/**
 * @brief Tests the allocation, property verification, and graceful deallocation of a Chunk.
 */
TEST_F(ChunkTest, AllocationAndDeallocation) {
    {
        size_t capacity = 1024;
        Chunk chunk(queue, capacity);
        
        // Sprawdź czy capacity i count są poprawne
        EXPECT_EQ(chunk.capacity(), capacity);
        EXPECT_EQ(chunk.count(), 0);

        // Sprawdź czy widok poprawnie odzwierciedla wskaźniki
        ChunkSoAView view = chunk.get_view();
        EXPECT_NE(view.d_is_dead, nullptr);
        EXPECT_NE(view.px, nullptr);
        EXPECT_EQ(view.capacity, capacity);
        EXPECT_EQ(view.count, 0);
    } 
    // Po wyjściu ze scope destruktor wywoła release(). 
    // W SYCL brak rzucania wyjątku i poprawne przejście oznacza brak problemów z pamięcią.
    SUCCEED();
}

/**
 * @brief Tests the move constructor of the Chunk class.
 */
TEST_F(ChunkTest, MoveConstructor) {
    size_t capacity = 512;

    Chunk chunk1(queue, capacity);
    ChunkSoAView view1 = chunk1.get_view();
    uint32_t* original_is_dead = view1.d_is_dead;

    // Przeniesienie konstruktorem przenoszącym
    Chunk chunk2(std::move(chunk1));

    EXPECT_EQ(chunk2.capacity(), capacity);
    
    // Sprawdź czy wskaźniki przeszły do chunk2
    ChunkSoAView view2 = chunk2.get_view();
    EXPECT_EQ(view2.d_is_dead, original_is_dead);

    // Sprawdź czy chunk1 został wyczyszczony (wskaźniki na nullptr przez std::exchange)
    ChunkSoAView view1_after = chunk1.get_view();
    EXPECT_EQ(view1_after.d_is_dead, nullptr);
}

/**
 * @brief Tests the move assignment operator of the Chunk class.
 */
TEST_F(ChunkTest, MoveAssignment) {
    Chunk chunk1(queue, 256);
    Chunk chunk2(queue, 128);

    ChunkSoAView view1 = chunk1.get_view();
    uint32_t* original_is_dead1 = view1.d_is_dead;

    // Przypisanie przenoszące
    chunk2 = std::move(chunk1);

    EXPECT_EQ(chunk2.capacity(), 256);
    EXPECT_EQ(chunk2.get_view().d_is_dead, original_is_dead1);
    EXPECT_EQ(chunk1.get_view().d_is_dead, nullptr);
}

/**
 * @brief Tests that `ChunkSoAView` accurately exposes all structural component pointers and updated metadata.
 */
TEST_F(ChunkTest, GetViewPointersMatch) {
    Chunk chunk(queue, 64);
    
    // Modyfikacja count i sprawdzenie czy view to odzwierciedla
    chunk.set_count(32);
    
    ChunkSoAView view = chunk.get_view();
    EXPECT_EQ(view.count, 32);
    EXPECT_EQ(view.capacity, 64);
    
    // Sprawdź czy wszystkie wskaźniki w view są różne od nullptr
    EXPECT_NE(view.px, nullptr);
    EXPECT_NE(view.py, nullptr);
    EXPECT_NE(view.pz, nullptr);
    EXPECT_NE(view.vx, nullptr);
    EXPECT_NE(view.vy, nullptr);
    EXPECT_NE(view.vz, nullptr);
    EXPECT_NE(view.d_dead_count, nullptr);
    EXPECT_NE(view.d_is_dead, nullptr);
}

/**
 * @test Verifies that marking the same particle multiple times for removal 
 *       does not duplicate its index in d_dead_indices or increment d_dead_count multiple times.
 */
TEST_F(ChunkTest, MarkDuplicateParticleSafety) {
    const size_t capacity = 100;
    Chunk chunk(queue, capacity);

    // Sztucznie ustawiamy 10 aktywnych cząstek w chunku
    chunk.set_count(10);

    auto view = chunk.get_view();

    // Zerujemy licznik na GPU
    queue.memset(view.d_dead_count, 0, sizeof(int32_t)).wait();

    // 1. Oznaczamy cząstkę 3 trzykrotnie oraz cząstkę 7 dwukrotnie
    view.mark_for_removal(queue, 3);
    view.mark_for_removal(queue, 3);
    view.mark_for_removal(queue, 3);

    view.mark_for_removal(queue, 7);
    view.mark_for_removal(queue, 7);

    queue.wait();

    // 2. Weryfikacja: Mimo 5 wywołań zarejestrowane powinny zostać tylko 2 unikalne cząstki
    int32_t h_dead_count = 0;
    queue.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 2);

    // 3. Odczytujemy bitmaskę z bufora na GPU i sprawdzamy, czy zaznaczone są dokładnie i wyłącznie cząstki 3 oraz 7
    size_t num_word_bits = (capacity + 31) / 32;
    std::vector<uint32_t> h_is_dead(num_word_bits, 0);
    queue.memcpy(h_is_dead.data(), view.d_is_dead, num_word_bits * sizeof(uint32_t)).wait();

    auto is_bit_set = [&](size_t idx) {
        size_t word_idx = idx / 32;
        uint32_t bit_mask = 1u << (idx % 32);
        return (h_is_dead[word_idx] & bit_mask) != 0;
    };

    EXPECT_TRUE(is_bit_set(3));
    EXPECT_TRUE(is_bit_set(7));

    // Upewniamy się, że pozostałe cząstki z zakresu 0-9 nie zostały dotknięte
    for (int i = 0; i < 10; ++i) {
        if (i == 3 || i == 7) continue;
        EXPECT_FALSE(is_bit_set(i));
    }
}

/**
 * @brief Tests particle marking for removal, verifying both the bitmask (d_is_dead_) and d_dead_indices_.
 */
TEST_F(ChunkTest, MarkForRemovalBitmaskAndIndices) {
    size_t capacity = 64;
    Chunk chunk(queue, capacity);
    
    // Ustawiamy np. 10 aktywnych cząstek w chunku
    size_t active_count = 10;
    chunk.set_count(active_count);

    ChunkSoAView view = chunk.get_view();

    // 1. Oznaczamy cząstki o indeksach 2 i 5 do usunięcia
    ASSERT_NO_THROW({
        view.mark_for_removal(queue, 2).wait();
        view.mark_for_removal(queue, 5).wait();
    });

    // 2. Próbujemy oznaczyć cząstkę 2 ponownie (duplikat - powinna zostać zignorowana)
    ASSERT_NO_THROW({
        view.mark_for_removal(queue, 2).wait();
    });

    // 3. Sprawdzamy licznik martwych cząstek (d_dead_count) na hoście
    int32_t h_dead_count = -1;
    queue.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 2); // Powinny być dokładnie 2 unikalne zgłoszenia

    // 4. Sprawdzamy bitmaskę (d_is_dead_) na hoście
    // Ponieważ d_is_dead_ to uint32_t*, dla capacity 64 potrzebujemy 2 słów (2 * 32 = 64 bity)
    size_t num_words = (capacity + 31) / 32;
    std::vector<uint32_t> h_dead_bits(num_words, 0);
    queue.memcpy(h_dead_bits.data(), view.d_is_dead, num_words * sizeof(uint32_t)).wait();

    // Weryfikacja bitów dla indeksów 2 i 5 (są w pierwszym słowie, word_idx = 0)
    EXPECT_TRUE((h_dead_bits[0] & (1u << 2)) != 0) << "Particle 2 bit should be set!";
    EXPECT_TRUE((h_dead_bits[0] & (1u << 5)) != 0) << "Particle 5 bit should be set!";
    
    // Weryfikacja, że inna cząstka (np. 3) NIE jest oznaczona
    EXPECT_FALSE((h_dead_bits[0] & (1u << 3)) != 0) << "Particle 3 bit should remain unset!";
}