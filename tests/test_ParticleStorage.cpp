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

    // --- 3. WERYFIKACJA REJESTRACJI MARTWYCH CZĄSTEK W VRAM ---

    // Sprawdzenie SpeciesID = 0
    {
        int32_t dead_count = 0;
        int32_t dead_idx = -1;
        q.memcpy(&dead_count, last_view_0.d_dead_count, sizeof(int32_t)).wait();
        q.memcpy(&dead_idx, last_view_0.d_dead_indices, sizeof(int32_t)).wait();

        EXPECT_EQ(dead_count, 1);
        EXPECT_EQ(dead_idx, static_cast<int32_t>(last_idx_0));
    }

    // Sprawdzenie SpeciesID = 1
    {
        int32_t dead_count = 0;
        std::vector<int32_t> dead_indices(2, -1);
        q.memcpy(&dead_count, first_view_1.d_dead_count, sizeof(int32_t)).wait();
        q.memcpy(dead_indices.data(), first_view_1.d_dead_indices, 2 * sizeof(int32_t)).wait();

        EXPECT_EQ(dead_count, 2);
        EXPECT_TRUE((dead_indices[0] == 0 && dead_indices[1] == 1) ||
                    (dead_indices[0] == 1 && dead_indices[1] == 0));
    }

    // Sprawdzenie SpeciesID = 2
    {
        int32_t dead_count = 0;
        std::vector<int32_t> dead_indices(2, -1);
        q.memcpy(&dead_count, second_view_2.d_dead_count, sizeof(int32_t)).wait();
        q.memcpy(dead_indices.data(), second_view_2.d_dead_indices, 2 * sizeof(int32_t)).wait();

        EXPECT_EQ(dead_count, 2);
        int32_t expected_first = static_cast<int32_t>(mid_idx_2);
        int32_t expected_second = static_cast<int32_t>(mid_idx_2 + 1);
        
        EXPECT_TRUE((dead_indices[0] == expected_first && dead_indices[1] == expected_second) ||
                    (dead_indices[0] == expected_second && dead_indices[1] == expected_first));
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

