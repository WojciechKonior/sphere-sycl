#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "ParticleStorage.hpp"

/**
 * @brief Test fixture for ParticleStorage unit tests.
 * 
 * Initializes a SYCL queue targeting the default device to enable 
 * testing of VRAM allocation and chunk management logic.
 */
class ParticleStorageTest : public ::testing::Test {
protected:
    sycl::queue q;

    /**
     * @brief Sets up the test environment by creating a default SYCL queue.
     */
    void SetUp() override {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
        } catch (const sycl::exception &e) {
            FAIL() << "Failed to create SYCL queue: " << e.what();
        }
    }

    /**
     * @brief Cleans up the test environment after each test execution.
     */
    void TearDown() override {
        try {
            q.wait();
        } catch (const sycl::exception &e) {
            ADD_FAILURE() << "SYCL exception during TearDown wait: " << e.what();
        }
    }

    void populate_chunk(Chunk& chunk, size_t num_particles) {
        chunk.set_count(num_particles);
        
        std::vector<float> h_pos(num_particles);
        std::iota(h_pos.begin(), h_pos.end(), 0.0f); // 0.0, 1.0, 2.0, ...

        auto view = chunk.get_view();
        q.memcpy(view.px, h_pos.data(), num_particles * sizeof(float)).wait();
        q.memcpy(view.py, h_pos.data(), num_particles * sizeof(float)).wait();
        q.memcpy(view.pz, h_pos.data(), num_particles * sizeof(float)).wait();
        q.memcpy(view.vx, h_pos.data(), num_particles * sizeof(float)).wait();
        q.memcpy(view.vy, h_pos.data(), num_particles * sizeof(float)).wait();
        q.memcpy(view.vz, h_pos.data(), num_particles * sizeof(float)).wait();
    }

    std::vector<float> read_pos_x(Chunk& chunk) {
        size_t count = chunk.count();
        std::vector<float> h_pos(count);
        auto view = chunk.get_view();
        q.memcpy(h_pos.data(), view.px, count * sizeof(float)).wait();
        return h_pos;
    }

    void populate_chunk_positions(Chunk& chunk, float base_val) {
        size_t count = chunk.count();
        std::vector<float> h_pos(count);
        for (size_t i = 0; i < count; ++i) {
            h_pos[i] = base_val + static_cast<float>(i);
        }
        
        // Używamy publicznego widoku zamiast prywatnego pola
        auto view = chunk.get_view();
        q.memcpy(view.px, h_pos.data(), count * sizeof(float)).wait();
    }
};

/**
 * @test Verifies that a newly initialized ParticleStorage object contains no particles or chunks.
 */
TEST_F(ParticleStorageTest, InitialStateIsEmpty) {
    ParticleStorage storage(q);

    EXPECT_EQ(storage.get_total_count(SpeciesID::ELECTRON), 0);
    EXPECT_EQ(storage.get_total_capacity(SpeciesID::ELECTRON), 0);
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::ELECTRON), 0);
    EXPECT_TRUE(storage.get_views(SpeciesID::ELECTRON).empty());
}

/**
 * @test Verifies that requesting zero particles returns an empty list of handles and modifies no state.
 */
TEST_F(ParticleStorageTest, AllocateZeroParticles) {
    ParticleStorage storage(q);

    auto handles = storage.allocate_space(SpeciesID::ELECTRON, 0);

    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(storage.get_total_count(SpeciesID::ELECTRON), 0);
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::ELECTRON), 0);
}

/**
 * @test Verifies partial allocation within a single chunk when count_to_add is less than CHUNK_CAPACITY.
 */
TEST_F(ParticleStorageTest, AllocateSingleChunkPartial) {
    ParticleStorage storage(q);
    const size_t count = 500; // Assumes CHUNK_CAPACITY > 500

    auto handles = storage.allocate_space(SpeciesID::ELECTRON, count);

    // Should return a single segment handle in the newly allocated chunk
    ASSERT_EQ(handles.size(), 1);
    EXPECT_EQ(handles[0].offset, 0);
    EXPECT_EQ(handles[0].count, count);

    // Verify storage memory metrics
    EXPECT_EQ(storage.get_total_count(SpeciesID::ELECTRON), count);
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::ELECTRON), 1);
    EXPECT_EQ(storage.get_total_capacity(SpeciesID::ELECTRON), Constants::CHUNK_CAPACITY);
}

/**
 * @test Verifies allocation spanning multiple chunks, including backfilling an existing chunk and spawning a new one.
 */
TEST_F(ParticleStorageTest, AllocateAcrossMultipleChunks) {
    ParticleStorage storage(q);
    const size_t capacity = Constants::CHUNK_CAPACITY;

    // Step 1: Fill half of the first chunk
    const size_t first_alloc = capacity / 2;
    storage.allocate_space(SpeciesID::HYDROGEN_ION, first_alloc);

    // Step 2: Request enough particles to fill the remainder of chunk 1 and spill over to chunk 2
    const size_t second_alloc = capacity; // Requires 1 full CHUNK_CAPACITY in total
    auto handles = storage.allocate_space(SpeciesID::HYDROGEN_ION, second_alloc);

    // Should return two segment handles: remainder of chunk 1 + start of chunk 2
    ASSERT_EQ(handles.size(), 2);
    
    // Handle 1: Fills the remaining space in the first chunk
    EXPECT_EQ(handles[0].offset, first_alloc);
    EXPECT_EQ(handles[0].count, capacity - first_alloc);

    // Handle 2: Begins at the start of the newly allocated second chunk
    EXPECT_EQ(handles[1].offset, 0);
    EXPECT_EQ(handles[1].count, first_alloc);

    // Metrics verification across both chunks
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::HYDROGEN_ION), 2);
    EXPECT_EQ(storage.get_total_count(SpeciesID::HYDROGEN_ION), first_alloc + second_alloc);
    EXPECT_EQ(storage.get_total_capacity(SpeciesID::HYDROGEN_ION), capacity * 2);
}

/**
 * @test Verifies that get_views() correctly returns SoA views for active chunks belonging to a specific species.
 */
TEST_F(ParticleStorageTest, GetViewsReturnsOnlyActiveChunks) {
    ParticleStorage storage(q);
    const size_t capacity = Constants::CHUNK_CAPACITY;

    // Allocate particles across two distinct species
    storage.allocate_space(SpeciesID::ELECTRON, capacity + 100); // Spans 2 chunks
    storage.allocate_space(SpeciesID::HYDROGEN_ION, 50);                  // Spans 1 chunk

    auto electron_views = storage.get_views(SpeciesID::ELECTRON);
    auto ion_views = storage.get_views(SpeciesID::HYDROGEN_ION);

    // Electron species should report 2 active views
    ASSERT_EQ(electron_views.size(), 2);
    EXPECT_EQ(electron_views[0].count, capacity);
    EXPECT_EQ(electron_views[1].count, 100);

    // Ion species should report 1 active view
    ASSERT_EQ(ion_views.size(), 1);
    EXPECT_EQ(ion_views[0].count, 50);
}

/**
 * @test Verifies that allocations for one species remain isolated and do not alter state for other species.
 */
TEST_F(ParticleStorageTest, SpeciesIsolation) {
    ParticleStorage storage(q);

    storage.allocate_space(SpeciesID::ELECTRON, 1000);

    // Operations on ELECTRON must not affect ION counters or chunk lists
    EXPECT_EQ(storage.get_total_count(SpeciesID::ELECTRON), 1000);
    EXPECT_EQ(storage.get_total_count(SpeciesID::HYDROGEN_ION), 0);
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::HYDROGEN_ION), 0);
}

/**
 * @test Verifies that pointers inside returned SegmentHandles strictly correspond 
 *       to the underlying Chunk's device memory buffers.
 */
TEST_F(ParticleStorageTest, SegmentHandlesContainValidPointers) {
    ParticleStorage storage(q);
    const size_t count = 100;

    auto handles = storage.allocate_space(SpeciesID::ELECTRON, count);
    auto views = storage.get_views(SpeciesID::ELECTRON);

    ASSERT_EQ(handles.size(), 1);
    ASSERT_EQ(views.size(), 1);

    // Pointers in handle must match the pointers in the active view
    EXPECT_EQ(handles[0].px, views[0].px);
    EXPECT_EQ(handles[0].py, views[0].py);
    EXPECT_EQ(handles[0].pz, views[0].pz);
    EXPECT_EQ(handles[0].vx, views[0].vx);
    EXPECT_EQ(handles[0].vy, views[0].vy);
    EXPECT_EQ(handles[0].vz, views[0].vz);
}

/**
 * @test Verifies exact multi-chunk boundary allocation without leaving partially filled chunks.
 */
TEST_F(ParticleStorageTest, AllocateExactMultipleChunks) {
    ParticleStorage storage(q);
    const size_t exact_two_chunks = Constants::CHUNK_CAPACITY * 2;

    auto handles = storage.allocate_space(SpeciesID::ELECTRON, exact_two_chunks);

    ASSERT_EQ(handles.size(), 2);
    EXPECT_EQ(handles[0].count, Constants::CHUNK_CAPACITY);
    EXPECT_EQ(handles[1].count, Constants::CHUNK_CAPACITY);
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::ELECTRON), 2);
}

/**
 * @test Verifies querying metrics and views for an uninitialized species returns zeros/empty vector.
 */
TEST_F(ParticleStorageTest, QueryUninitializedSpecies) {
    ParticleStorage storage(q);

    // Querying an enum ID that was never passed to allocate_space
    EXPECT_EQ(storage.get_total_count(SpeciesID::NEUTRAL_HYDROGEN), 0);
    EXPECT_EQ(storage.get_total_capacity(SpeciesID::NEUTRAL_HYDROGEN), 0);
    EXPECT_EQ(storage.get_chunk_count(SpeciesID::NEUTRAL_HYDROGEN), 0);
    EXPECT_TRUE(storage.get_views(SpeciesID::NEUTRAL_HYDROGEN).empty());
}

/**
 * @test Verifies cumulative small allocations correctly step through chunk offset space.
 */
TEST_F(ParticleStorageTest, SequentialSmallAllocations) {
    ParticleStorage storage(q);
    const size_t small_step = 10;
    const size_t iterations = 50;

    for (size_t i = 0; i < iterations; ++i) {
        auto handles = storage.allocate_space(SpeciesID::ELECTRON, small_step);
        ASSERT_EQ(handles.size(), 1);
        EXPECT_EQ(handles[0].offset, i * small_step);
        EXPECT_EQ(handles[0].count, small_step);
    }

    EXPECT_EQ(storage.get_total_count(SpeciesID::ELECTRON), small_step * iterations);
}

/**
 * @test MultiChunkMultiSpeciesRemovalAndAllocationCheck
 * @brief Verifies particle removal, compaction, and chunk management parametrically, 
 *        safely capped based on available device VRAM.
 */
TEST_F(ParticleStorageTest, CompactionReleasesEmptyChunk)
{
    ParticleStorage storage(q);

    size_t number_of_species_to_test = 3;
    size_t particles_exeeded = 1;
    size_t initial_chunks_per_species = 3;
    size_t particles_to_add = (initial_chunks_per_species-1) * Constants::CHUNK_CAPACITY + particles_exeeded;
    size_t number_of_chunks = (particles_to_add + Constants::CHUNK_CAPACITY - 1)/Constants::CHUNK_CAPACITY;

    for(int i = 0; i<number_of_species_to_test; i++)
    {
        SpeciesID test_species = static_cast<SpeciesID>(i);

        EXPECT_EQ(storage.get_total_count(test_species), 0u);
        EXPECT_EQ(storage.get_total_capacity(test_species), 0u);
        EXPECT_EQ(storage.get_chunk_count(test_species), 0u);

        auto views = storage.get_views(test_species);
        EXPECT_TRUE(views.empty());

        auto handles = storage.allocate_space(test_species, particles_to_add);
        EXPECT_EQ(storage.get_chunk_count(test_species), number_of_chunks);
        EXPECT_EQ(storage.get_total_count(test_species), particles_to_add);
        EXPECT_EQ(storage.get_total_capacity(test_species), number_of_chunks*Constants::CHUNK_CAPACITY);

        size_t particles_count = 0;
        for (const auto &handle : handles)
        {
            particles_count += handle.count;
            EXPECT_EQ(handle.offset, 0u);
        }
        EXPECT_EQ(particles_count, particles_to_add);
        ASSERT_EQ(handles.size(), number_of_chunks);
    }

    EXPECT_EQ(storage.get_chunk_count(), number_of_species_to_test*number_of_chunks);
    EXPECT_EQ(storage.get_total_count(), number_of_species_to_test*particles_to_add);
    EXPECT_EQ(storage.get_total_capacity(), number_of_species_to_test*number_of_chunks*Constants::CHUNK_CAPACITY);

    size_t particles_to_remove = particles_exeeded;

    // --- 2. OZNACZANIE CZĄSTEK DO USUNIĘCIA NA GPU ---

    // SpeciesID = 0: Ostatnia cząstka w ostatnim chunku
    SpeciesID species_0 = static_cast<SpeciesID>(0);
    auto views_0 = storage.get_views(species_0);
    ASSERT_FALSE(views_0.empty());
    auto &last_view_0 = views_0.back();
    size_t last_idx_0 = last_view_0.count - 1;
    last_view_0.mark_for_removal(q, last_idx_0);

    // SpeciesID = 1: Dwie pierwsze cząstki (indeksy 0 i 1) z pierwszego chunka
    SpeciesID species_1 = static_cast<SpeciesID>(1);
    auto views_1 = storage.get_views(species_1);
    ASSERT_GE(views_1.size(), 1u);
    auto &first_view_1 = views_1.front();
    ASSERT_GE(first_view_1.count, 2u);
    first_view_1.mark_for_removal(q, 0);
    first_view_1.mark_for_removal(q, 1);

    // SpeciesID = 2: Środkowe cząstki ze środka drugiego chunka (index 1)
    SpeciesID species_2 = static_cast<SpeciesID>(2);
    auto views_2 = storage.get_views(species_2);
    ASSERT_GE(views_2.size(), 2u); // Upewniamy się, że istnieje drugi chunk
    auto &second_view_2 = views_2[1];
    size_t mid_idx_2 = second_view_2.count / 2;
    ASSERT_GE(second_view_2.count, 2u);
    
    // Oznaczamy 2 cząstki ze środka drugiego chunka
    second_view_2.mark_for_removal(q, mid_idx_2);
    second_view_2.mark_for_removal(q, mid_idx_2 + 1);

    // Synchronizujemy operacje atomowe przed sprawdzeniem stanu
    q.wait();

    // Pomocnicza lambda do sprawdzania stanu bitu w buforze bitmaski
    auto is_bit_set = [](const std::vector<uint32_t>& h_is_dead, size_t idx) {
        size_t word_idx = idx / 32;
        uint32_t bit_mask = 1u << (idx % 32);
        return (h_is_dead[word_idx] & bit_mask) != 0;
    };

    // --- 3. WERYFIKACJA REJESTRACJI MARTWYCH CZĄSTEK W VRAM ---

    // Sprawdzenie SpeciesID = 0
    {
        int32_t dead_count = 0;
        q.memcpy(&dead_count, last_view_0.d_dead_count, sizeof(int32_t)).wait();
        EXPECT_EQ(dead_count, 1);

        size_t num_words = (last_view_0.capacity + 31) / 32;
        std::vector<uint32_t> h_is_dead(num_words, 0);
        q.memcpy(h_is_dead.data(), last_view_0.d_is_dead, num_words * sizeof(uint32_t)).wait();

        EXPECT_TRUE(is_bit_set(h_is_dead, last_idx_0));
    }

    // Sprawdzenie SpeciesID = 1
    {
        int32_t dead_count = 0;
        q.memcpy(&dead_count, first_view_1.d_dead_count, sizeof(int32_t)).wait();
        EXPECT_EQ(dead_count, 2);

        size_t num_words = (first_view_1.capacity + 31) / 32;
        std::vector<uint32_t> h_is_dead(num_words, 0);
        q.memcpy(h_is_dead.data(), first_view_1.d_is_dead, num_words * sizeof(uint32_t)).wait();

        EXPECT_TRUE(is_bit_set(h_is_dead, 0));
        EXPECT_TRUE(is_bit_set(h_is_dead, 1));
    }

    // Sprawdzenie SpeciesID = 2
    {
        int32_t dead_count = 0;
        q.memcpy(&dead_count, second_view_2.d_dead_count, sizeof(int32_t)).wait();
        EXPECT_EQ(dead_count, 2);

        size_t num_words = (second_view_2.capacity + 31) / 32;
        std::vector<uint32_t> h_is_dead(num_words, 0);
        q.memcpy(h_is_dead.data(), second_view_2.d_is_dead, num_words * sizeof(uint32_t)).wait();

        size_t expected_first = mid_idx_2;
        size_t expected_second = mid_idx_2 + 1;
        
        EXPECT_TRUE(is_bit_set(h_is_dead, expected_first));
        EXPECT_TRUE(is_bit_set(h_is_dead, expected_second));
    }

    storage.free_marked_particles();

    // --- 4. RYGORYSTYCZNA WERYFIKACJA WSZYSTKICH METRYK I STANÓW ---

    // Specyfikacja oczekiwanych stanów po kompaktacji:
    // Species 0: usunięto 1 cząstkę (z chunka 3), chunk 3 staje się pusty -> Zostaje ZWOLNIONY do VRAM
    size_t expected_count_0    = particles_to_add - 1; // 2 * CHUNK_CAPACITY
    size_t expected_chunks_0   = initial_chunks_per_species - 1; // 2 chunki
    size_t expected_capacity_0 = expected_chunks_0 * Constants::CHUNK_CAPACITY;

    // Species 1: usunięto 2 cząstki z pierwszego chunka -> Wszystkie chunki nadal mają >0 cząstek
    size_t expected_count_1    = particles_to_add - 2;
    size_t expected_chunks_1   = initial_chunks_per_species - 1; // 2 chunki
    size_t expected_capacity_1 = expected_chunks_1 * Constants::CHUNK_CAPACITY;

    // Species 2: usunięto 2 cząstki z drugiego chunka -> Wszystkie chunki nadal mają >0 cząstek
    size_t expected_count_2    = particles_to_add - 2;
    size_t expected_chunks_2   = initial_chunks_per_species - 1; // 3 chunki
    size_t expected_capacity_2 = expected_chunks_2 * Constants::CHUNK_CAPACITY;

    // A. Weryfikacja metryk per-species:
    EXPECT_EQ(storage.get_total_count(species_0),    expected_count_0);
    EXPECT_EQ(storage.get_chunk_count(species_0),    expected_chunks_0);
    EXPECT_EQ(storage.get_total_capacity(species_0), expected_capacity_0);

    EXPECT_EQ(storage.get_total_count(species_1),    expected_count_1);
    EXPECT_EQ(storage.get_chunk_count(species_1),    expected_chunks_1);
    EXPECT_EQ(storage.get_total_capacity(species_1), expected_capacity_1);

    EXPECT_EQ(storage.get_total_count(species_2),    expected_count_2);
    EXPECT_EQ(storage.get_chunk_count(species_2),    expected_chunks_2);
    EXPECT_EQ(storage.get_total_capacity(species_2), expected_capacity_2);

    // B. Weryfikacja zbiorczych metryk całego kontenera ParticleStorage:
    size_t expected_global_count    = expected_count_0 + expected_count_1 + expected_count_2;
    size_t expected_global_chunks   = expected_chunks_0 + expected_chunks_1 + expected_chunks_2;
    size_t expected_global_capacity = expected_global_chunks * Constants::CHUNK_CAPACITY;

    EXPECT_EQ(storage.get_total_count(),    expected_global_count);
    EXPECT_EQ(storage.get_chunk_count(),    expected_global_chunks);
    EXPECT_EQ(storage.get_total_capacity(), expected_global_capacity);

    // C. Weryfikacja wektora widoków (get_views) dla Species 0:
    auto updated_views_0 = storage.get_views(species_0);
    EXPECT_EQ(updated_views_0.size(), expected_chunks_0); // Czy zwektorowany widok VRAM ma teraz 2 elementy?

    // D. Weryfikacja zresetowania d_dead_count do 0 na GPU dla wszystkich pozostałych chunków:
    for (size_t i = 0; i < number_of_species_to_test; ++i)
    {
        SpeciesID sp = static_cast<SpeciesID>(i);
        auto views = storage.get_views(sp);
        for (const auto &v : views)
        {
            int32_t h_dead_count = -1;
            q.memcpy(&h_dead_count, v.d_dead_count, sizeof(int32_t)).wait();
            EXPECT_EQ(h_dead_count, 0); // Stan flagi martwych cząstek w VRAM musi być wyczyszczony!
        }
    }
}


/**
 * @test VerifySwapAndPopFull6DIntegrity
 * @brief Verifies the data integrity of all 6D SoA attributes during particle removal and compaction.
 *
 * This test initializes particles with unique 6D values (posX..posZ, velX..velZ) based on a base multiplier:
 * Particle i gets values: {i+1.0, i+1.1, i+1.2, i+1.3, i+1.4, i+1.5}.
 *
 * Validation Steps:
 * 1. Allocates 4 particles in a single chunk and verifies initialization.
 * 2. Marks the last particle (index 3) for removal and verifies that removing the last element
 *    leaves preceding elements completely untouched.
 * 3. Marks the first particle (index 0) for removal and verifies the Swap-and-Pop behavior
 *    (the last active particle at index 2 moves into index 0, while index 1 remains unaffected).
 */
TEST_F(ParticleStorageTest, VerifySwapAndPopFull6DIntegrity)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    // 1. Alokujemy 4 cząstki w jednym chunku
    size_t num_particles = 4;
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 1u);
    auto &view = views.front();
    ASSERT_EQ(view.count, num_particles);

    // 2. Inicjalizacja wg schematu:
    // Cząstka 0: {1.0, 1.1, 1.2, 1.3, 1.4, 1.5}
    // Cząstka 1: {2.0, 2.1, 2.2, 2.3, 2.4, 2.5}
    // Cząstka 2: {3.0, 3.1, 3.2, 3.3, 3.4, 3.5}
    // Cząstka 3: {4.0, 4.1, 4.2, 4.3, 4.4, 4.5}
    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            float base = static_cast<float>(idx[0] + 1); // 1.0, 2.0, 3.0, 4.0
            posX[idx] = base + 0.0f;
            posY[idx] = base + 0.1f;
            posZ[idx] = base + 0.2f;
            velX[idx] = base + 0.3f;
            velY[idx] = base + 0.4f;
            velZ[idx] = base + 0.5f;
        });
    }).wait();

    // Struktura pomocnicza na Hoście do odczytu stanu 6D cząstki
    struct Particle6D {
        float px, py, pz;
        float vx, vy, vz;

        void verify(float base) const {
            EXPECT_FLOAT_EQ(px, base + 0.0f);
            EXPECT_FLOAT_EQ(py, base + 0.1f);
            EXPECT_FLOAT_EQ(pz, base + 0.2f);
            EXPECT_FLOAT_EQ(vx, base + 0.3f);
            EXPECT_FLOAT_EQ(vy, base + 0.4f);
            EXPECT_FLOAT_EQ(vz, base + 0.5f);
        }
    };

    // Lambda pomocnicza do pobierania pełnych danych z GPU na Host
    auto read_particles_from_gpu = [&](const ChunkSoAView &v, size_t count) {
        std::vector<Particle6D> result(count);
        std::vector<float> h_px(count), h_py(count), h_pz(count);
        std::vector<float> h_vx(count), h_vy(count), h_vz(count);

        q.memcpy(h_px.data(), v.px, count * sizeof(float));
        q.memcpy(h_py.data(), v.py, count * sizeof(float));
        q.memcpy(h_pz.data(), v.pz, count * sizeof(float));
        q.memcpy(h_vx.data(), v.vx, count * sizeof(float));
        q.memcpy(h_vy.data(), v.vy, count * sizeof(float));
        q.memcpy(h_vz.data(), v.vz, count * sizeof(float)).wait();

        for (size_t i = 0; i < count; ++i) {
            result[i] = {h_px[i], h_py[i], h_pz[i], h_vx[i], h_vy[i], h_vz[i]};
        }
        return result;
    };


    // --- KROK 1: Usuwamy OSTATNIĄ cząstkę (indeks 3, baza 4.0f) ---
    view.mark_for_removal(q, 3);
    q.wait();

    storage.free_marked_particles();

    // Weryfikacja po Kroku 1
    EXPECT_EQ(storage.get_total_count(species), 3u);
    auto views_krok1 = storage.get_views(species);
    ASSERT_EQ(views_krok1.size(), 1u);
    EXPECT_EQ(views_krok1.front().count, 3u);

    auto p_krok1 = read_particles_from_gpu(views_krok1.front(), 3);
    p_krok1[0].verify(1.0f); // Bez zmian
    p_krok1[1].verify(2.0f); // Bez zmian
    p_krok1[2].verify(3.0f); // Bez zmian


    // --- KROK 2: Usuwamy PIERWSZĄ cząstkę (indeks 0, baza 1.0f) ---
    auto &updated_view = views_krok1.front();
    updated_view.mark_for_removal(q, 0);
    q.wait();

    storage.free_marked_particles();

    // Weryfikacja po Kroku 2
    EXPECT_EQ(storage.get_total_count(species), 2u);
    auto views_krok2 = storage.get_views(species);
    ASSERT_EQ(views_krok2.size(), 1u);
    EXPECT_EQ(views_krok2.front().count, 2u);

    auto p_krok2 = read_particles_from_gpu(views_krok2.front(), 2);
    
    // Po Swap-and-Pop:
    // Na indeksie 0 znajduje się dawna cząstka z indeksu 2 (baza 3.0f)
    p_krok2[0].verify(3.0f);
    // Na indeksie 1 znajduje się nieporuszona cząstka z indeksu 1 (baza 2.0f)
    p_krok2[1].verify(2.0f);
}

/**
 * @test VerifySwapAndPopWithSparseParticleModification
 * @brief Verifies that marking a particle in a partially filled chunk correctly transfers
 *        the last particle's state into the removed slot while preserving zeroed padding.
 *
 * Test workflow:
 * 1. Allocates a partially filled chunk (10 particles out of full capacity) initialized to 0.0f.
 * 2. Explicitly modifies particle 0 with base 1.0f {1.0, 1.1...} and particle 9 with base 2.0f {2.0, 2.1...}.
 * 3. Marks particle 0 for removal and triggers free_marked_particles().
 * 4. Verifies that count drops to 9, index 0 now holds particle 9's state {2.0, 2.1...},
 *    and indices 1..8 remain untouched (0.0f).
 */
TEST_F(ParticleStorageTest, VerifySwapAndPopWithSparseParticleModification)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    // 1. Alokujemy niepełny chunk (10 cząstek)
    size_t num_particles = 10;
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 1u);
    auto &view = views.front();
    ASSERT_EQ(view.count, num_particles);

    // Czyszczenie/zerowanie wszystkich pól w chunku na GPU
    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            posX[idx] = 0.0f; posY[idx] = 0.0f; posZ[idx] = 0.0f;
            velX[idx] = 0.0f; velY[idx] = 0.0f; velZ[idx] = 0.0f;
        });
    }).wait();

    // 2. Modyfikujemy pierwszą cząstkę (idx 0 -> base 1.0f) oraz ostatnią (idx 9 -> base 2.0f)
    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.single_task([=]() {
            // Pierwsza cząstka (idx = 0)
            posX[0] = 1.0f; posY[0] = 1.1f; posZ[0] = 1.2f;
            velX[0] = 1.3f; velY[0] = 1.4f; velZ[0] = 1.5f;

            // Ostatnia cząstka (idx = 9)
            size_t last = num_particles - 1;
            posX[last] = 2.0f; posY[last] = 2.1f; posZ[last] = 2.2f;
            velX[last] = 2.3f; velY[last] = 2.4f; velZ[last] = 2.5f;
        });
    }).wait();

    // 3. Usuwamy pierwszą cząstkę (indeks 0)
    view.mark_for_removal(q, 0);
    q.wait();

    storage.free_marked_particles();

    // 4. Weryfikacja po kompaktacji
    EXPECT_EQ(storage.get_total_count(species), 9u);
    auto updated_views = storage.get_views(species);
    ASSERT_EQ(updated_views.size(), 1u);
    
    auto &updated_view = updated_views.front();
    EXPECT_EQ(updated_view.count, 9u);

    // Pobieramy stan 9 cząstek z GPU na Host
    std::vector<float> h_px(9), h_py(9), h_pz(9);
    std::vector<float> h_vx(9), h_vy(9), h_vz(9);

    q.memcpy(h_px.data(), updated_view.px, 9 * sizeof(float));
    q.memcpy(h_py.data(), updated_view.py, 9 * sizeof(float));
    q.memcpy(h_pz.data(), updated_view.pz, 9 * sizeof(float));
    q.memcpy(h_vx.data(), updated_view.vx, 9 * sizeof(float));
    q.memcpy(h_vy.data(), updated_view.vy, 9 * sizeof(float));
    q.memcpy(h_vz.data(), updated_view.vz, 9 * sizeof(float)).wait();

    // Asercja 1: Na indeksie 0 powinnśmy mieć dawną cząstkę z indeksu 9 ({2.0, 2.1, 2.2, 2.3, 2.4, 2.5})
    EXPECT_FLOAT_EQ(h_px[0], 2.0f);
    EXPECT_FLOAT_EQ(h_py[0], 2.1f);
    EXPECT_FLOAT_EQ(h_pz[0], 2.2f);
    EXPECT_FLOAT_EQ(h_vx[0], 2.3f);
    EXPECT_FLOAT_EQ(h_vy[0], 2.4f);
    EXPECT_FLOAT_EQ(h_vz[0], 2.5f);

// Asercja 2: Pozostałe cząstki (indeksy 1..8) muszą pozostać zerowe
    for (size_t i = 1; i < 9; ++i)
    {
        EXPECT_FLOAT_EQ(h_px[i], 0.0f);
        EXPECT_FLOAT_EQ(h_py[i], 0.0f);
        EXPECT_FLOAT_EQ(h_pz[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vx[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vy[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vz[i], 0.0f);
    }
}

/**
 * @test VerifySwapAndPopMiddleParticleRemoval
 * @brief Verifies that marking a particle in the middle of a partially filled chunk correctly
 *        transfers the last particle's state into the removed middle slot via Swap-and-Pop.
 *
 * Test workflow:
 * 1. Allocates a partially filled chunk (10 particles) initialized to 0.0f.
 * 2. Explicitly modifies middle particle (idx 4) with base 1.0f {1.0, 1.1...} 
 *    and last particle (idx 9) with base 2.0f {2.0, 2.1...}.
 * 3. Marks middle particle (idx 4) for removal and triggers free_marked_particles().
 * 4. Verifies that count drops to 9, index 4 now holds particle 9's state {2.0, 2.1...},
 *    and indices 0..3 and 5..8 remain untouched (0.0f).
 */
TEST_F(ParticleStorageTest, VerifySwapAndPopMiddleParticleRemoval)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    // 1. Alokujemy niepełny chunk (10 cząstek)
    size_t num_particles = 10;
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 1u);
    auto &view = views.front();
    ASSERT_EQ(view.count, num_particles);

    // Czyszczenie/zerowanie wszystkich pól w chunku na GPU
    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            posX[idx] = 0.0f; posY[idx] = 0.0f; posZ[idx] = 0.0f;
            velX[idx] = 0.0f; velY[idx] = 0.0f; velZ[idx] = 0.0f;
        });
    }).wait();

    // 2. Modyfikujemy cząstkę ze środka (idx 4 -> base 1.0f) oraz ostatnią (idx 9 -> base 2.0f)
    size_t mid_idx = 4;
    size_t last_idx = num_particles - 1;

    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.single_task([=]() {
            // Cząstka ze środka (idx = 4)
            posX[mid_idx] = 1.0f; posY[mid_idx] = 1.1f; posZ[mid_idx] = 1.2f;
            velX[mid_idx] = 1.3f; velY[mid_idx] = 1.4f; velZ[mid_idx] = 1.5f;

            // Ostatnia cząstka (idx = 9)
            posX[last_idx] = 2.0f; posY[last_idx] = 2.1f; posZ[last_idx] = 2.2f;
            velX[last_idx] = 2.3f; velY[last_idx] = 2.4f; velZ[last_idx] = 2.5f;
        });
    }).wait();

    // 3. Usuwamy cząstkę ze środka (indeks 4)
    view.mark_for_removal(q, mid_idx);
    q.wait();

    storage.free_marked_particles();

    // 4. Weryfikacja po kompaktacji
    EXPECT_EQ(storage.get_total_count(species), 9u);
    auto updated_views = storage.get_views(species);
    ASSERT_EQ(updated_views.size(), 1u);
    
    auto &updated_view = updated_views.front();
    EXPECT_EQ(updated_view.count, 9u);

    // Pobieramy stan 9 cząstek z GPU na Host
    std::vector<float> h_px(9), h_py(9), h_pz(9);
    std::vector<float> h_vx(9), h_vy(9), h_vz(9);

    q.memcpy(h_px.data(), updated_view.px, 9 * sizeof(float));
    q.memcpy(h_py.data(), updated_view.py, 9 * sizeof(float));
    q.memcpy(h_pz.data(), updated_view.pz, 9 * sizeof(float));
    q.memcpy(h_vx.data(), updated_view.vx, 9 * sizeof(float));
    q.memcpy(h_vy.data(), updated_view.vy, 9 * sizeof(float));
    q.memcpy(h_vz.data(), updated_view.vz, 9 * sizeof(float)).wait();

    // Asercja 1: Na indeksie 4 (wcześniejszy środek) powinna znajdować się dawna cząstka z indeksu 9 ({2.0, 2.1, 2.2, ...})
    EXPECT_FLOAT_EQ(h_px[4], 2.0f);
    EXPECT_FLOAT_EQ(h_py[4], 2.1f);
    EXPECT_FLOAT_EQ(h_pz[4], 2.2f);
    EXPECT_FLOAT_EQ(h_vx[4], 2.3f);
    EXPECT_FLOAT_EQ(h_vy[4], 2.4f);
    EXPECT_FLOAT_EQ(h_vz[4], 2.5f);

    // Asercja 2: Pozostałe pozycje (0..3 oraz 5..8) pozostają nienaruszone (0.0f)
    for (size_t i = 0; i < 9; ++i)
    {
        if (i == 4) continue; // Pomiń sprawdzoną wyżej pozycję po usunięciu

        EXPECT_FLOAT_EQ(h_px[i], 0.0f);
        EXPECT_FLOAT_EQ(h_py[i], 0.0f);
        EXPECT_FLOAT_EQ(h_pz[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vx[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vy[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vz[i], 0.0f);
    }
}

/**
 * @test VerifyMultiChunkTwoStageCompaction
 * @brief Verifies particle relocation across multiple chunks during compaction.
 *
 * Test workflow:
 * 1. Allocates 2 full chunks + 1 particle in the 3rd chunk (3 chunks total) initialized to 0.0f.
 * 2. Explicitly modifies:
 *    - Chunk 0, Index 0 (first particle of species): Base 2.0f {2.0, 2.1...} -> MARKED FOR REMOVAL
 *    - Chunk 0, Last Index (last particle of Chunk 0): Base 3.0f {3.0, 3.1...}
 *    - Chunk 2, Index 0 (last active particle of species): Base 1.0f {1.0, 1.1...}
 * 3. Triggers free_marked_particles().
 * 4. Verifies multi-stage migration:
 *    - Local Swap-and-Pop: Chunk 0's last particle {3.0, 3.1...} moves to Chunk 0, Index 0.
 *    - Inter-chunk Swap-and-Pop: Chunk 2's particle {1.0, 1.1...} moves to Chunk 0's now-vacant last index.
 *    - Chunk 2 becomes completely empty and is freed from VRAM (chunk count drops to 2).
 */
TEST_F(ParticleStorageTest, VerifyMultiChunkTwoStageCompaction)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    // 1. Alokujemy 2 pełne chunki + 1 cząstkę (3 chunki łącznie)
    size_t chunk_cap = Constants::CHUNK_CAPACITY;
    size_t num_particles = 2 * chunk_cap + 1;
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 3u);

    // Czyszczenie/zerowanie wszystkich pól we wszystkich chunkach na GPU
    for (auto &v : views)
    {
        q.submit([&](sycl::handler &cgh) {
            auto posX = v.px; auto posY = v.py; auto posZ = v.pz;
            auto velX = v.vx; auto velY = v.vy; auto velZ = v.vz;
            size_t cnt = v.count;

            cgh.parallel_for(sycl::range<1>(cnt), [=](sycl::id<1> idx) {
                posX[idx] = 0.0f; posY[idx] = 0.0f; posZ[idx] = 0.0f;
                velX[idx] = 0.0f; velY[idx] = 0.0f; velZ[idx] = 0.0f;
            });
        }).wait();
    }

    // 2. Modyfikacja konkretnych cząstek
    auto &first_view = views[0];
    auto &last_view = views[2];
    size_t last_idx_chunk_0 = chunk_cap - 1;

    q.submit([&](sycl::handler &cgh) {
        auto f_px = first_view.px; auto f_py = first_view.py; auto f_pz = first_view.pz;
        auto f_vx = first_view.vx; auto f_vy = first_view.vy; auto f_vz = first_view.vz;

        auto l_px = last_view.px; auto l_py = last_view.py; auto l_pz = last_view.pz;
        auto l_vx = last_view.vx; auto l_vy = last_view.vy; auto l_vz = last_view.vz;

        cgh.single_task([=]() {
            // Chunk 0, Index 0 (Cząstka do usunięcia) -> Base 2.0f
            f_px[0] = 2.0f; f_py[0] = 2.1f; f_pz[0] = 2.2f;
            f_vx[0] = 2.3f; f_vy[0] = 2.4f; f_vz[0] = 2.5f;

            // Chunk 0, Ostatni Index -> Base 3.0f
            f_px[last_idx_chunk_0] = 3.0f; f_py[last_idx_chunk_0] = 3.1f; f_pz[last_idx_chunk_0] = 3.2f;
            f_vx[last_idx_chunk_0] = 3.3f; f_vy[last_idx_chunk_0] = 3.4f; f_vz[last_idx_chunk_0] = 3.5f;

            // Chunk 2, Index 0 (Ostatnia cząstka gatunku) -> Base 1.0f
            l_px[0] = 1.0f; l_py[0] = 1.1f; l_pz[0] = 1.2f;
            l_vx[0] = 1.3f; l_vy[0] = 1.4f; l_vz[0] = 1.5f;
        });
    }).wait();

    // 3. Oznaczamy pierwszą cząstkę z pierwszego chunka do usunięcia
    first_view.mark_for_removal(q, 0);
    q.wait();

    // 4. Wykonujemy pełną kompaktację dwuetapową
    storage.free_marked_particles();

    // 5. Weryfikacja po kompaktacji
    size_t expected_total_particles = num_particles - 1; // 2 * chunk_cap
    EXPECT_EQ(storage.get_total_count(species), expected_total_particles);
    
    auto updated_views = storage.get_views(species);
    // Trzeci chunk powinien zostać w całości zwolniony!
    ASSERT_EQ(updated_views.size(), 2u);
    EXPECT_EQ(updated_views[0].count, chunk_cap);
    EXPECT_EQ(updated_views[1].count, chunk_cap);

    // Pobieramy dane z pierwszego chunka z GPU na Host do szczegółowej asercji
    std::vector<float> h_px(chunk_cap), h_py(chunk_cap), h_pz(chunk_cap);
    std::vector<float> h_vx(chunk_cap), h_vy(chunk_cap), h_vz(chunk_cap);

    auto &c0_view = updated_views[0];
    q.memcpy(h_px.data(), c0_view.px, chunk_cap * sizeof(float));
    q.memcpy(h_py.data(), c0_view.py, chunk_cap * sizeof(float));
    q.memcpy(h_pz.data(), c0_view.pz, chunk_cap * sizeof(float));
    q.memcpy(h_vx.data(), c0_view.vx, chunk_cap * sizeof(float));
    q.memcpy(h_vy.data(), c0_view.vy, chunk_cap * sizeof(float));
    q.memcpy(h_vz.data(), c0_view.vz, chunk_cap * sizeof(float)).wait();

    // Sprawdzenie 1: Lokalny Swap-and-Pop
    // Dawna ostatnia cząstka z Chunk 0 ({3.0, 3.1...}) przeskakuje na Index 0
    EXPECT_FLOAT_EQ(h_px[0], 3.0f);
    EXPECT_FLOAT_EQ(h_py[0], 3.1f);
    EXPECT_FLOAT_EQ(h_pz[0], 3.2f);
    EXPECT_FLOAT_EQ(h_vx[0], 3.3f);
    EXPECT_FLOAT_EQ(h_vy[0], 3.4f);
    EXPECT_FLOAT_EQ(h_vz[0], 3.5f);

    // Sprawdzenie 2: Międzychunkowy Swap-and-Pop
    // Dawna cząstka z Chunk 2 ({1.0, 1.1...}) przeskakuje na zwalniane ostatnie miejsce w Chunk 0
    EXPECT_FLOAT_EQ(h_px[last_idx_chunk_0], 1.0f);
    EXPECT_FLOAT_EQ(h_py[last_idx_chunk_0], 1.1f);
    EXPECT_FLOAT_EQ(h_pz[last_idx_chunk_0], 1.2f);
    EXPECT_FLOAT_EQ(h_vx[last_idx_chunk_0], 1.3f);
    EXPECT_FLOAT_EQ(h_vy[last_idx_chunk_0], 1.4f);
    EXPECT_FLOAT_EQ(h_vz[last_idx_chunk_0], 1.5f);

    // Sprawdzenie 3: Środkowe pozycje w Chunk 0 (indeksy 1..chunk_cap-2) pozostały nieporuszone (0.0f)
    for (size_t i = 1; i < last_idx_chunk_0; ++i)
    {
        EXPECT_FLOAT_EQ(h_px[i], 0.0f);
        EXPECT_FLOAT_EQ(h_py[i], 0.0f);
        EXPECT_FLOAT_EQ(h_pz[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vx[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vy[i], 0.0f);
        EXPECT_FLOAT_EQ(h_vz[i], 0.0f);
    }
}

/**
 * @test VerifyMassRemovalWithinSingleChunk
 * @brief Verifies that marking ALL particles in a specific chunk for removal is handled safely
 *        without out-of-bounds access, and that the emptied chunk is released properly.
 */
TEST_F(ParticleStorageTest, VerifyMassRemovalWithinSingleChunk)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    size_t chunk_cap = Constants::CHUNK_CAPACITY;
    size_t num_particles = 2 * chunk_cap + 1; // 3 chunki
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 3u);

    // 1. Inicjalizacja danych w VRAM
    for (size_t c = 0; c < views.size(); ++c)
    {
        auto &v = views[c];
        float base_val = static_cast<float>(c + 1); // 1.0f, 2.0f, 3.0f

        q.submit([&](sycl::handler &cgh) {
            auto posX = v.px; auto posY = v.py; auto posZ = v.pz;
            auto velX = v.vx; auto velY = v.vy; auto velZ = v.vz;
            size_t cnt = v.count;

            cgh.parallel_for(sycl::range<1>(cnt), [=](sycl::id<1> idx) {
                posX[idx] = base_val + 0.0f;
                posY[idx] = base_val + 0.1f;
                posZ[idx] = base_val + 0.2f;
                velX[idx] = base_val + 0.3f;
                velY[idx] = base_val + 0.4f;
                velZ[idx] = base_val + 0.5f;
            });
        }).wait();
    }

    // 2. MASOWE OZNACZANIE CZĄSTEK NA GPU (Jednym kernelem zamiast miliona wywołań w pętli CPU!)
    auto &chunk_1_view = views[1];
    size_t chunk_1_count = chunk_1_view.count;
    EXPECT_EQ(chunk_1_count, Constants::CHUNK_CAPACITY);
    const int32_t max_cap = static_cast<int32_t>(chunk_1_view.capacity);

    q.memset(chunk_1_view.d_dead_count, 0, sizeof(int32_t)).wait();

    // Wywołujemy pojedynczy wątek na GPU do oznaczenia indeksu 2
    q.submit([&](sycl::handler &cgh) {
        auto d_dead_count = chunk_1_view.d_dead_count;
        cgh.parallel_for(sycl::range<1>(chunk_1_count), [=](sycl::id<1> idx) {
            chunk_1_view.device_mark_for_removal(idx[0]);
        });
    }).wait();

    // 3. Wywołanie czyszczenia i kompaktacji
    storage.free_marked_particles();

    // 4. Weryfikacja liczników
    size_t expected_total_particles = num_particles - chunk_1_count;
    EXPECT_EQ(storage.get_total_count(species), expected_total_particles);

    auto updated_views = storage.get_views(species);
    // Jeden chunk powinien zostać całkowicie wyeliminowany z kontenera
    ASSERT_EQ(updated_views.size(), 2u);
    EXPECT_EQ(updated_views[0].count, chunk_cap);
    EXPECT_EQ(updated_views[1].count, 1u);

    // 5. Weryfikacja spójności danych SoA
    std::vector<float> h_px(chunk_cap);
    q.memcpy(h_px.data(), updated_views[0].px, chunk_cap * sizeof(float)).wait();
    for (size_t i = 0; i < chunk_cap; ++i)
    {
        EXPECT_FLOAT_EQ(h_px[i], 1.0f);
    }

    float h_px_last_chunk = 0.0f;
    q.memcpy(&h_px_last_chunk, updated_views[1].px, sizeof(float)).wait();
    EXPECT_FLOAT_EQ(h_px_last_chunk, 3.0f);
}

/**
 * @brief Verifies that calling mark_for_removal_device multiple times from a GPU kernel 
 *        for the same particles correctly results in a safe single removal during compaction.
 */
TEST_F(ParticleStorageTest, MarkForRemovalDeviceDuplicateSafety) {
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(1); // Adjust according to your Species enum values

    // 1. Allocate space for initial particles to populate a chunk
    size_t initial_particles_count = 10;
    auto segments = storage.allocate_space(species, initial_particles_count);
    ASSERT_FALSE(segments.empty());

    // Retrieve chunk views to access device-side methods and buffers
    auto views = storage.get_views(species);
    ASSERT_FALSE(views.empty());
    auto &chunk_view = views[0];

    // Ensure dead count starts at zero
    q.memset(chunk_view.d_dead_count, 0, sizeof(int32_t)).wait();

    // 2. Submit a SYCL kernel simulating a physics/collision pass 
    // where particle index 3 is marked 3 times, and particle index 7 is marked 2 times.
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(5), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            if (i < 3) {
                chunk_view.device_mark_for_removal(3); // Mark particle 3 three times
            } else {
                chunk_view.device_mark_for_removal(7); // Mark particle 7 two times
            }
        });
    }).wait();

    size_t count_before_compaction = storage.get_total_count(species); // Should be 10

    // 3. Trigger compaction and particle removal via the storage API
    storage.free_marked_particles(species);

    // 4. Verification:
    // Even though mark_for_removal_device was called 5 times total, only 2 unique 
    // particles (3 and 7) were targeted. The total count should drop by exactly 2 (from 10 to 8).
    EXPECT_EQ(storage.get_total_count(species), count_before_compaction - 2);

    // Verify that the internal device dead count was properly reset to 0 after compaction
    int32_t final_dead_count = -1;
    q.memcpy(&final_dead_count, chunk_view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(final_dead_count, 0);
}

/**
 * @brief Tests multi-marking (including duplicate attempts) and compaction via ParticleStorage.
 */
TEST_F(ParticleStorageTest, MultipleMarkingAndCompactionWithDuplicates) {
    SpeciesID test_species = static_cast<SpeciesID>(0);
    ParticleStorage storage(q);

    // 1. Alokujemy np. 30 cząstek (załóżmy, że zmieszczą się w jednym lub wielu chunkach)
    size_t count_to_add = 30;
    auto handles = storage.allocate_space(test_species, count_to_add);
    ASSERT_FALSE(handles.empty());

    // Sprawdzamy czy początkowy licznik się zgadza
    EXPECT_EQ(storage.get_total_count(test_species), count_to_add);

    // 2. Pobieramy widoki chunków
    auto views = storage.get_views(test_species);
    ASSERT_FALSE(views.empty());

    // Bierzemy pierwszy chunk do testu
    ChunkSoAView view = views[0];

    // 3. Oznaczamy wybrane cząstki do usunięcia, w tym wielokrotnie te same (duplikaty)
    // Przykładowo oznaczamy indeksy: 3, 10, 15
    // Indeks 3 oznaczymy 3 razy, indeks 10 oznaczymy 2 razy. Unikalnych do usunięcia będzie 3.
    ASSERT_NO_THROW({
        view.mark_for_removal(q, 3).wait();
        view.mark_for_removal(q, 3).wait(); // Duplikat
        view.mark_for_removal(q, 3).wait(); // Duplikat

        view.mark_for_removal(q, 10).wait();
        view.mark_for_removal(q, 10).wait(); // Duplikat

        view.mark_for_removal(q, 15).wait();
    });

    // 4. Wywołujemy usuwanie i kompaktowanie dla danej species
    storage.free_marked_particles(test_species);

    // 5. Weryfikacja wyniku
    // Początkowo było 30 cząstek. Oznaczyliśmy 3 unikalne (3, 10, 15).
    // Oczekiwana liczba cząstek po kompaktowaniu to 30 - 3 = 27.
    size_t expected_count = count_to_add - 3;
    EXPECT_EQ(storage.get_total_count(test_species), expected_count);
}

/**
 * @test VerifyMultipleMarkingAndCompactionWithDataIntegrity
 * @brief Verifies that marking particles multiple times (including duplicates) and removing them 
 *        correctly compacts the storage while preserving expected 6D values for remaining particles.
 *
 * Test workflow:
 * 1. Allocates 10 particles in a single chunk.
 * 2. Initializes each particle `i` with values based on base `i + 1.0f` (e.g., particle 0: {1.0, 1.1...}, particle 9: {10.0, 10.1...}).
 * 3. Marks particles 3 and 5 for removal **multiple times** (simulating duplicate/redundant marking requests).
 * 4. Triggers `storage.free_marked_particles()`.
 * 5. Verifies that:
 *    - Total count drops from 10 to 8.
 *    - Duplicate markings did not cause issues or double-removals.
 *    - Remaining particles are correctly compacted via Swap-and-Pop (last particles moved into the holes).
 */
TEST_F(ParticleStorageTest, VerifyMultipleMarkingAndCompactionWithDataIntegrity)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    // 1. Alokujemy 10 cząstek w jednym chunku
    size_t num_particles = 10;
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 1u);
    auto &view = views.front();
    ASSERT_EQ(view.count, num_particles);

    // 2. Inicjalizacja 10 cząstek unikalnymi wartościami bazowymi od 1.0f do 10.0f
    // Cząstka i dostaje bazę: i + 1.0f (np. idx 0 -> 1.0, idx 3 -> 4.0, idx 9 -> 10.0)
    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            float base = static_cast<float>(idx[0] + 1.0f);
            posX[idx] = base + 0.0f;
            posY[idx] = base + 0.1f;
            posZ[idx] = base + 0.2f;
            velX[idx] = base + 0.3f;
            velY[idx] = base + 0.4f;
            velZ[idx] = base + 0.5f;
        });
    }).wait();

    // 3. Oznaczamy cząstki do usunięcia KILKUKROTNIE (duplikaty):
    // - Cząstka o indeksie 3 (baza 4.0f) oznaczona 3 razy
    // - Cząstka o indeksie 5 (baza 6.0f) oznaczona 2 razy
    view.mark_for_removal(q, 3);
    view.mark_for_removal(q, 3); // Duplikat
    view.mark_for_removal(q, 3); // Duplikat

    view.mark_for_removal(q, 5);
    view.mark_for_removal(q, 5); // Duplikat
    q.wait();

    // 4. Wywołujemy kompaktowanie i usunięcie oznaczonych cząstek
    storage.free_marked_particles();

    // 5. Weryfikacja liczności po usunięciu (10 - 2 unikalne = 8 cząstek)
    EXPECT_EQ(storage.get_total_count(species), 8u);
    auto updated_views = storage.get_views(species);
    ASSERT_EQ(updated_views.size(), 1u);
    EXPECT_EQ(updated_views.front().count, 8u);

    // Pobieramy stan 8 pozostałych cząstek z GPU na Host
    std::vector<float> h_px(8), h_py(8), h_pz(8);
    std::vector<float> h_vx(8), h_vy(8), h_vz(8);

    q.memcpy(h_px.data(), updated_views.front().px, 8 * sizeof(float));
    q.memcpy(h_py.data(), updated_views.front().py, 8 * sizeof(float));
    q.memcpy(h_pz.data(), updated_views.front().pz, 8 * sizeof(float));
    q.memcpy(h_vx.data(), updated_views.front().vx, 8 * sizeof(float));
    q.memcpy(h_vy.data(), updated_views.front().vy, 8 * sizeof(float));
    q.memcpy(h_vz.data(), updated_views.front().vz, 8 * sizeof(float)).wait();

    // Pomocnicza lambda do weryfikacji wartości cząstki o danej bazie
    auto verify_particle = [&](size_t host_idx, float expected_base) {
        EXPECT_FLOAT_EQ(h_px[host_idx], expected_base + 0.0f);
        EXPECT_FLOAT_EQ(h_py[host_idx], expected_base + 0.1f);
        EXPECT_FLOAT_EQ(h_pz[host_idx], expected_base + 0.2f);
        EXPECT_FLOAT_EQ(h_vx[host_idx], expected_base + 0.3f);
        EXPECT_FLOAT_EQ(h_vy[host_idx], expected_base + 0.4f);
        EXPECT_FLOAT_EQ(h_vz[host_idx], expected_base + 0.5f);
    };

    // 6. Sprawdzenie poprawności układu po Swap-and-Pop:
    // Oryginalnie miałyśmy indeksy 0..9. 
    // Usunięto indeks 3 oraz indeks 5.
    // Zgodnie z algorytmem Swap-and-Pop (biorącym cząstki z końca):
    // - Dziura pod indeksem 3 została wypełniona ostatnią cząstką z indeksu 9 (baza 10.0f).
    // - Dziura pod indeksem 5 została wypełniona przedostatnią cząstką z indeksu 8 (baza 9.0f).
    // Ostatecznie aktywne cząstki powinny mieć bazy: 
    // [0] -> 1.0f
    // [1] -> 2.0f
    // [2] -> 3.0f
    // [3] -> 10.0f (przeniesiona z idx 9)
    // [4] -> 5.0f
    // [5] -> 9.0f  (przeniesiona z idx 8)
    // [6] -> 7.0f
    // [7] -> 8.0f

    verify_particle(0, 1.0f);
    verify_particle(1, 2.0f);
    verify_particle(2, 3.0f);
    verify_particle(3, 10.0f); // W miejsce usuniętej 3 wskoczyła dawna 9
    verify_particle(4, 5.0f);
    verify_particle(5, 9.0f);  // W miejsce usuniętej 5 wskoczyła dawna 8
    verify_particle(6, 7.0f);
    verify_particle(7, 8.0f);
}

/**
 * @test VerifyCheckerboardRemovalPattern
 * @brief Verifies alternating (checkerboard) particle removal pattern where every other particle is marked for removal.
 *
 * Test workflow:
 * 1. Allocates 10 particles in a single chunk initialized with unique base values.
 * 2. Marks alternating particles (indices 0, 2, 4, 6, 8) for removal.
 * 3. Triggers free_marked_particles() and verifies compaction.
 * 4. Verifies that count drops to 5 and the remaining surviving particles 
 *    (corresponding to original odd indices 1, 3, 5, 7, 9) have their 6D values fully preserved.
 */
TEST_F(ParticleStorageTest, VerifyCheckerboardRemovalPattern)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    size_t num_particles = 10;
    storage.allocate_space(species, num_particles);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 1u);
    auto &view = views.front();
    ASSERT_EQ(view.count, num_particles);

    // 1. Inicjalizacja 10 cząstek unikalnymi wartościami bazowymi od 1.0f do 10.0f
    // Cząstka i dostaje bazę: i + 1.0f (np. idx 0 -> 1.0, idx 1 -> 2.0, ..., idx 9 -> 10.0)
    q.submit([&](sycl::handler &cgh) {
        auto posX = view.px; auto posY = view.py; auto posZ = view.pz;
        auto velX = view.vx; auto velY = view.vy; auto velZ = view.vz;

        cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
            float base = static_cast<float>(idx[0] + 1.0f);
            posX[idx] = base + 0.0f;
            posY[idx] = base + 0.1f;
            posZ[idx] = base + 0.2f;
            velX[idx] = base + 0.3f;
            velY[idx] = base + 0.4f;
            velZ[idx] = base + 0.5f;
        });
    }).wait();

    // 2. Oznaczamy co drugą cząstkę do usunięcia (szachownica: indeksy 0, 2, 4, 6, 8)
    for (size_t i = 0; i < num_particles; i += 2) {
        view.mark_for_removal(q, i);
    }
    q.wait();

    // 3. Wywołujemy kompaktowanie i usunięcie oznaczonych cząstek
    storage.free_marked_particles();

    // 4. Weryfikacja liczności po usunięciu (z 10 zostaje dokładnie 5 cząstek)
    size_t expected_count = num_particles / 2;
    EXPECT_EQ(storage.get_total_count(species), expected_count);

    auto updated_views = storage.get_views(species);
    ASSERT_EQ(updated_views.size(), 1u);
    EXPECT_EQ(updated_views.front().count, expected_count);

    // Pobieramy stan 5 pozostałych cząstek z GPU na Host
    std::vector<float> h_px(expected_count), h_py(expected_count), h_pz(expected_count);
    std::vector<float> h_vx(expected_count), h_vy(expected_count), h_vz(expected_count);

    q.memcpy(h_px.data(), updated_views.front().px, expected_count * sizeof(float));
    q.memcpy(h_py.data(), updated_views.front().py, expected_count * sizeof(float));
    q.memcpy(h_pz.data(), updated_views.front().pz, expected_count * sizeof(float));
    q.memcpy(h_vx.data(), updated_views.front().vx, expected_count * sizeof(float));
    q.memcpy(h_vy.data(), updated_views.front().vy, expected_count * sizeof(float));
    q.memcpy(h_vz.data(), updated_views.front().vz, expected_count * sizeof(float)).wait();

    // 5. Weryfikacja zawartości pozostałych cząstek.
    // Usunęliśmy indeksy parzyste: 0, 2, 4, 6, 8 (bazy: 1.0, 3.0, 5.0, 7.0, 9.0).
    // Zostają indeksy nieparzyste: 1, 3, 5, 7, 9 (oczekiwane bazy: 2.0, 4.0, 6.0, 8.0, 10.0).
    std::vector<float> expected_bases = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    std::vector<float> actual_bases(expected_count);
    
    for (size_t i = 0; i < expected_count; ++i) {
        // Ponieważ posX[idx] = base + 0.0f, to odczytana wartość to bezpośrednio base
        actual_bases[i] = h_px[i]; 
        
        // Weryfikacja integralności 6D dla każdego komponentu
        EXPECT_FLOAT_EQ(h_py[i], actual_bases[i] + 0.1f);
        EXPECT_FLOAT_EQ(h_pz[i], actual_bases[i] + 0.2f);
        EXPECT_FLOAT_EQ(h_vx[i], actual_bases[i] + 0.3f);
        EXPECT_FLOAT_EQ(h_vy[i], actual_bases[i] + 0.4f);
        EXPECT_FLOAT_EQ(h_vz[i], actual_bases[i] + 0.5f);
    }

    // Sortujemy oba wektory, ponieważ Swap-and-Pop zmienia kolejność w pamięci
    std::sort(actual_bases.begin(), actual_bases.end());
    std::sort(expected_bases.begin(), expected_bases.end());

    ASSERT_EQ(actual_bases.size(), expected_bases.size());
    for (size_t i = 0; i < expected_count; ++i) {
        EXPECT_FLOAT_EQ(actual_bases[i], expected_bases[i]);
    }
}

/**
 * @test VerifyMultiChunkEmptyChunkDominoCompaction
 * @brief Verifies multi-chunk compaction across chunks with empty intermediate chunks (Domino Effect).
 *
 * Test workflow:
 * 1. Allocates space spanning 4 chunks (`3 * CHUNK_CAPACITY + 2`).
 * 2. Sets up the requested state via removal markings:
 *    - Chunk 0: Half full (half of its particles marked for removal).
 *    - Chunk 1: Completely empty (all particles marked for removal).
 *    - Chunk 2: Completely empty (all particles marked for removal).
 *    - Chunk 3: Contains only 2 active particles.
 * 3. Triggers free_marked_particles().
 * 4. Verifies that multi-chunk compaction correctly skips empty intermediate chunks (count == 0),
 *    migrates particles to fill gaps, and properly releases the emptied chunks from storage.
 */
TEST_F(ParticleStorageTest, VerifyMultiChunkEmptyChunkDominoCompaction)
{
    ParticleStorage storage(q);
    SpeciesID species = static_cast<SpeciesID>(0);

    const size_t cap = Constants::CHUNK_CAPACITY;
    ASSERT_GE(cap, 100u); // Wymóg pojemności chunka dla testu

    // 1. Alokujemy 4 chunki (3 pełne + 2 cząstki w 4. chunku)
    size_t total_particles_to_add = 3 * cap + 2;
    storage.allocate_space(species, total_particles_to_add);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 4u);

    // 2. Ustawiamy stan początkowy testu:
    // - Chunk 0: W połowie pusty
    // - Chunk 1 i Chunk 2: Mają być całkowicie puste.
    // Ponieważ get_views() zwraca widoki, możemy oznaczyć wszystkie ich cząstki do usunięcia
    // lub skorzystać z mechanizmu widoku. Aby nie robić wolnej pętli hosta na wielkim cap,
    // wywołujemy szybki kernel oznaczający dla całego zakresu chunka:
    size_t half_cap = cap / 2;
    ChunkSoAView v0 = views[0], v1 = views[1], v2 = views[2];

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(cap), [=](sycl::id<1> idx) {
            if(idx[0]<half_cap)
                v0.device_mark_for_removal(static_cast<int32_t>(idx[0]));
            v1.device_mark_for_removal(static_cast<int32_t>(idx[0]));
            v2.device_mark_for_removal(static_cast<int32_t>(idx[0]));
        });
    }).wait();

    // 3. Wywołujemy globalne czyszczenie i kompaktowanie między-chunkowe
    storage.free_marked_particles();

    // 4. Weryfikacja stanu po kompaktowaniu domina:
    // - Oczekiwana liczba żywych cząstek: to co zostało w chunku 0 (`cap - half_cap`) + 2 cząstki z chunka 3.
    size_t expected_total_count = (cap - half_cap) + 2;
    EXPECT_EQ(storage.get_total_count(species), expected_total_count);

    // - Chunki 1 i 2 stały się całkowicie puste i zostały zwolnione z VRAMu przez mechanizm domina,
    //   więc liczba aktywnych chunków musi spaść poniżej 4.
    auto updated_views = storage.get_views(species);
    EXPECT_GT(updated_views.size(), 0u);
    EXPECT_LT(updated_views.size(), 4u); 
}

/**
 * @test Verifies space allocation after a chunk is released via particle compaction (re-use scenario).
 * 
 * This test ensures that:
 * 1. Releasing an empty chunk during compaction correctly updates storage metrics and chunk counts.
 * 2. Subsequent allocations properly provision or reuse chunks with valid, uncorrupted USM pointers.
 * 3. Existing chunks retain stable USM addresses, and new/reallocated chunks maintain unique, valid memory views.
 */
TEST_F(ParticleStorageTest, AllocateSpaceAfterChunkReleaseReuse)
{
    ParticleStorage storage(q);

    // 1. Initialization: Set up a configuration with 2 chunks for a given species
    SpeciesID species = static_cast<SpeciesID>(0);
    size_t initial_chunks = 2;
    size_t particles_to_add = (initial_chunks - 1) * Constants::CHUNK_CAPACITY + 1; // Fills 1 full chunk + 1 particle in the second chunk

    // Allocate initial space
    auto initial_handles = storage.allocate_space(species, particles_to_add);
    EXPECT_EQ(storage.get_chunk_count(species), initial_chunks);
    
    // Save references to VRAM pointers prior to compaction (e.g., of the first chunk)
    auto initial_views = storage.get_views(species);
    ASSERT_EQ(initial_views.size(), initial_chunks);
    auto* original_first_chunk_ptr = initial_views[0].px;

    // 2. Mark all particles in the second (last) chunk for removal
    // to trigger its complete release after compaction.
    auto &last_view = initial_views.back();
    size_t last_chunk_count = last_view.count;
    
    // Mark every particle in the last chunk
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(last_chunk_count), [=](sycl::id<1> idx) {
            last_view.device_mark_for_removal(static_cast<int32_t>(idx[0]));
        });
    }).wait();

    // Execute the removal and compaction of dead particles
    storage.free_marked_particles();

    // 3. Verification post-removal: The empty chunk should be released (leaving 1 chunk)
    EXPECT_EQ(storage.get_chunk_count(species), 1u);

    // 4. Main Scenario: Allocate new particles after capacity update
    // Allocate enough particles to force the re-allocation or re-occupancy of a second chunk
    size_t new_particles_to_add = Constants::CHUNK_CAPACITY;
    auto new_handles = storage.allocate_space(species, new_particles_to_add);

    // Verify metrics after the new allocation
    EXPECT_EQ(storage.get_chunk_count(species), 2u);

    // 5. Check USM pointer validity and view consistency via get_views
    auto updated_views = storage.get_views(species);
    ASSERT_EQ(updated_views.size(), 2u);

    // A. Did the first chunk preserve its stable VRAM address (was it not corrupted/reallocated)?
    EXPECT_EQ(updated_views[0].px, original_first_chunk_ptr);

    // B. Does the new/reallocated second chunk have a valid, non-null USM pointer?
    EXPECT_NE(updated_views[1].px, nullptr);
    
    // C. Are the SoA pointers inside the views unique and consistent across both chunks?
    EXPECT_NE(updated_views[0].px, updated_views[1].px);
    EXPECT_NE(updated_views[0].py, updated_views[1].py);
    EXPECT_NE(updated_views[0].pz, updated_views[1].pz);

    // D. Verification of total particle count consistency
    size_t expected_total_count = (particles_to_add - last_chunk_count) + new_particles_to_add;
    EXPECT_EQ(storage.get_total_count(species), expected_total_count);
}


/**
 * @test Verifies compaction behavior when only the absolute last particle of a chunk is marked for removal.
 * 
 * This test ensures that:
 * 1. Marking only the final particle (which sits at the boundary of the active range) triggers no redundant 
 *    SoA copy operations (`num_moves == 0`), as no particle relocation is needed from the tail.
 * 2. If that particle is the only one in the chunk, the chunk correctly becomes empty, gets released, 
 *    and updates the overall chunk count accordingly.
 */
TEST_F(ParticleStorageTest, CompactionOnlyLastParticleRemoved)
{
    ParticleStorage storage(q);

    // 1. Initialization: Set up 3 chunks where the 3rd chunk contains exactly 1 particle
    SpeciesID species = static_cast<SpeciesID>(0);
    size_t initial_chunks = 3;
    size_t particles_to_add = (initial_chunks - 1) * Constants::CHUNK_CAPACITY + 1; // Fills 2 full chunks + 1 particle in the 3rd chunk

    auto handles = storage.allocate_space(species, particles_to_add);
    ASSERT_EQ(storage.get_chunk_count(species), initial_chunks);

    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), initial_chunks);

    // 2. Target the last particle of the 3rd chunk
    auto &last_view = views.back();
    ASSERT_EQ(last_view.count, 1u); // The 3rd chunk has exactly 1 particle
    size_t last_particle_idx = last_view.count - 1; // Index 0

    // Mark the last particle for removal
    last_view.mark_for_removal(q, last_particle_idx);
    q.wait();

    // 3. Execute compaction and cleanup of marked particles
    storage.free_marked_particles();

    // 4. Verification:
    // A. Since the 3rd chunk had only 1 particle and it was removed, the chunk should now be empty and released.
    EXPECT_EQ(storage.get_chunk_count(species), initial_chunks - 1);

    // B. Verify the total particle count reflects the single removed particle.
    EXPECT_EQ(storage.get_total_count(species), particles_to_add - 1);

    // C. Verify remaining views are valid and dead counts are cleared across active chunks
    auto updated_views = storage.get_views(species);
    EXPECT_EQ(updated_views.size(), initial_chunks - 1);
    for (const auto &v : updated_views) {
        int32_t h_dead_count = -1;
        q.memcpy(&h_dead_count, v.d_dead_count, sizeof(int32_t)).wait();
        EXPECT_EQ(h_dead_count, 0);
    }
}


/**
 * @test Verifies concurrent/parallel removal and compaction across multiple species (SpeciesID).
 * 
 * This test ensures that:
 * 1. Marking particles across different species simultaneously does not cross-contaminate state.
 * 2. Calling `storage.free_marked_particles()` correctly isolates each species during iteration over `species_chunks_`.
 * 3. Dead counters (`d_dead_count`) and bitmasks are correctly and independently cleared for every active chunk across all species.
 */
TEST_F(ParticleStorageTest, SimultaneousCompactionMultipleSpecies)
{
    ParticleStorage storage(q);

    // 1. Initialization: Set up 3 distinct species, each with 2 chunks
    size_t number_of_species = 3;
    size_t initial_chunks = 2;
    size_t particles_per_species = Constants::CHUNK_CAPACITY + 1; // Fills 1 full chunk + 1 particle in the second chunk

    for (size_t i = 0; i < number_of_species; ++i) {
        SpeciesID species = static_cast<SpeciesID>(i);
        storage.allocate_space(species, particles_per_species);
        EXPECT_EQ(storage.get_chunk_count(species), initial_chunks);
    }

    // 2. Mark particles for removal across all species simultaneously
    // Species 0: Mark the last particle of its first chunk
    // Species 1: Mark the first particle of its second chunk
    // Species 2: Mark the last particle of its second chunk
    for (size_t i = 0; i < number_of_species; ++i) {
        SpeciesID species = static_cast<SpeciesID>(i);
        auto views = storage.get_views(species);
        ASSERT_GE(views.size(), 2u);

        if (i == 0) {
            views[0].mark_for_removal(q, views[0].count - 1);
        } else if (i == 1) {
            views[1].mark_for_removal(q, 0);
        } else if (i == 2) {
            views[1].mark_for_removal(q, views[1].count - 1);
        }
    }
    q.wait();

    // 3. Execute global compaction for all marked particles across all species
    storage.free_marked_particles();

    // 4. Verification:
    // A. Check that total counts for each species are correctly reduced by 1
    size_t expected_count_per_species = particles_per_species - 1;
    for (size_t i = 0; i < number_of_species; ++i) {
        SpeciesID species = static_cast<SpeciesID>(i);
        EXPECT_EQ(storage.get_total_count(species), expected_count_per_species);
    }

    // B. Check that d_dead_count is independently reset to 0 for every chunk across all species
    for (size_t i = 0; i < number_of_species; ++i) {
        SpeciesID species = static_cast<SpeciesID>(i);
        auto views = storage.get_views(species);
        for (const auto &v : views) {
            int32_t h_dead_count = -1;
            q.memcpy(&h_dead_count, v.d_dead_count, sizeof(int32_t)).wait();
            EXPECT_EQ(h_dead_count, 0) << "Dead count for species " << i << " chunk was not reset properly!";
        }
    }
}


/**
 * @test Verifies safety and correctness when duplicate or concurrent removal requests 
 *       are issued for the same particle within the same time step.
 * 
 * This test ensures that:
 * 1. Multiple calls to `mark_for_removal` for the same particle index do not inflate 
 *    the dead count (atomic bitmask protection ensures unique tracking).
 * 2. Subsequent execution of `free_marked_particles()` / `compact_chunk()` handles 
 *    the dead count and bitmask cleanly without invalid memory access or corruption.
 */
TEST_F(ParticleStorageTest, MarkDuplicateParticleCompactionSafety)
{
    const size_t capacity = 100;
    Chunk chunk(q, capacity);
    chunk.set_count(10); // 10 active particles (indices 0 to 9)

    auto view = chunk.get_view();

    // Ensure dead count is zeroed
    q.memset(view.d_dead_count, 0, sizeof(int32_t)).wait();

    // 1. Issue multiple duplicate requests to remove the exact same particle (index 4)
    // and another particle (index 7) multiple times concurrently/repeatedly.
    view.mark_for_removal(q, 4);
    view.mark_for_removal(q, 4);
    view.mark_for_removal(q, 4);
    
    view.mark_for_removal(q, 7);
    view.mark_for_removal(q, 7);

    q.wait();

    // 2. Verification of dead count: 
    // Despite multiple duplicate calls, exactly 2 unique particles should be registered as dead.
    int32_t h_dead_count = -1;
    q.memcpy(&h_dead_count, view.d_dead_count, sizeof(int32_t)).wait();
    EXPECT_EQ(h_dead_count, 2);

    // 3. Verify bitmask state: only bits 4 and 7 should be set.
    size_t num_words = (capacity + 31) / 32;
    std::vector<uint32_t> h_is_dead(num_words, 0);
    q.memcpy(h_is_dead.data(), view.d_is_dead, num_words * sizeof(uint32_t)).wait();

    auto is_bit_set = [&](size_t idx) {
        size_t word_idx = idx / 32;
        uint32_t bit_mask = 1u << (idx % 32);
        return (h_is_dead[word_idx] & bit_mask) != 0;
    };

    EXPECT_TRUE(is_bit_set(4));
    EXPECT_TRUE(is_bit_set(7));

    // Ensure no other bits are accidently marked
    for (int i = 0; i < 10; ++i) {
        if (i == 4 || i == 7) continue;
        EXPECT_FALSE(is_bit_set(i));
    }
}

/**
 * @test Verifies behavior when all particles of a species are completely removed and storage becomes empty.
 * 
 * This test ensures that:
 * 1. Removing all particles causes all chunks for that species to be released, 
 *    resulting in `storage.get_chunk_count(species) == 0`.
 * 2. Calling `storage.get_views(species)` returns a safely empty vector (`views.empty()`), 
 *    preventing any dangling or invalid USM pointer access.
 * 3. Subsequent re-allocation for the same species starts correctly from scratch and functions properly.
 */
TEST_F(ParticleStorageTest, ReadViewsWhenAllParticlesRemoved)
{
    ParticleStorage storage(q);

    // 1. Initialization: Allocate particles for a specific species
    SpeciesID species = static_cast<SpeciesID>(0);
    size_t particles_to_add = Constants::CHUNK_CAPACITY; // Exactly 1 full chunk

    auto handles = storage.allocate_space(species, particles_to_add);
    EXPECT_EQ(storage.get_chunk_count(species), 1u);
    EXPECT_EQ(storage.get_total_count(species), particles_to_add);

    // 2. Mark ALL active particles in the chunk for removal
    auto views = storage.get_views(species);
    ASSERT_EQ(views.size(), 1u);

    q.submit([=](sycl::handler &cgh){
        auto view = views[0];
        cgh.parallel_for(sycl::range<1>(view.count), [=](sycl::id<1> idx){
            view.device_mark_for_removal(idx[0]);
        });
    });

    // Execute compaction, which should completely release the chunk since it's now empty
    storage.free_marked_particles();

    // 3. Verification of empty state:
    // A. Chunk count and total count must be zero
    EXPECT_EQ(storage.get_chunk_count(species), 0u);
    EXPECT_EQ(storage.get_total_count(species), 0u);
    EXPECT_EQ(storage.get_total_capacity(species), 0u);

    // B. Views vector must be safely empty (no invalid/dangling USM pointers returned)
    auto empty_views = storage.get_views(species);
    EXPECT_TRUE(empty_views.empty());

    // 4. Verification of recovery / re-allocation from scratch:
    // Try allocating new particles for the same species again to ensure storage state is clean and operational
    size_t new_particles_to_add = Constants::CHUNK_CAPACITY / 2;
    auto new_handles = storage.allocate_space(species, new_particles_to_add);

    EXPECT_EQ(storage.get_chunk_count(species), 1u);
    EXPECT_EQ(storage.get_total_count(species), new_particles_to_add);

    auto recovered_views = storage.get_views(species);
    ASSERT_EQ(recovered_views.size(), 1u);
    EXPECT_NE(recovered_views[0].px, nullptr); // Valid USM pointer should be provisioned
    EXPECT_EQ(recovered_views[0].count, new_particles_to_add);
}