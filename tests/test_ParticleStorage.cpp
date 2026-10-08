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
 * @test Verifies compact_chunk behavior when a single particle in the middle is removed.
 * Last particle should be moved to the removed particle's slot.
 */
TEST_F(ParticleStorageTest, SingleParticleRemovalMiddle) {
    ParticleStorage storage(q);
    SpeciesID species_e = static_cast<SpeciesID>(1);

    // Utwórz 1 chunk z 5 cząstkami: [0.0, 1.0, 2.0, 3.0, 4.0]
    auto& chunk = storage.add_chunk(species_e, 10);
    populate_chunk(chunk, 5);

    // Oznacz cząstkę o indeksie 1 (wartość 1.0) do usunięcia na GPU
    auto view = chunk.get_view();
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            view.mark_for_removal(1);
        });
    }).wait();

    // Wykonaj kompaktowanie
    storage.free_marked_particles(species_e);

    // Oczekujemy 4 cząstek: [0.0, 4.0, 2.0, 3.0] (ostatnia cząstka 4.0 trafiła na indeks 1)
    EXPECT_EQ(chunk.count(), 4);

    auto result = read_pos_x(chunk);
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f); // Podmieniona cząstka z końca
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 3.0f);
}

/**
 * @test Verifies removing the last particle in a chunk (Edge case).
 * No Swap-and-Pop move required, counter should simply decrement.
 */
TEST_F(ParticleStorageTest, RemoveLastParticleOnly) {
    ParticleStorage storage(q);
    SpeciesID species_e = static_cast<SpeciesID>(1);

    auto& chunk = storage.add_chunk(species_e, 10);
    populate_chunk(chunk, 5); // [0.0, 1.0, 2.0, 3.0, 4.0]

    // Oznacz ostatnią cząstkę (indeks 4)
    auto view = chunk.get_view();
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            view.mark_for_removal(4);
        });
    }).wait();

    storage.free_marked_particles(species_e);

    EXPECT_EQ(chunk.count(), 4);

    auto result = read_pos_x(chunk);
    std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f};
    EXPECT_EQ(result, expected);
}

/**
 * @test Verifies removing multiple particles, including both middle and boundary particles.
 */
TEST_F(ParticleStorageTest, MultipleParticleRemovalsComplex) {
    ParticleStorage storage(q);
    SpeciesID species_e = static_cast<SpeciesID>(1);

    // Utwórz 1 chunk z 6 cząstkami: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    auto& chunk = storage.add_chunk(species_e, 10);
    populate_chunk(chunk, 6);

    // Usuwamy indeksy 1, 3 oraz 5 (ostatni)
    auto view = chunk.get_view();
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(6), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            if (id == 1 || id == 3 || id == 5) {
                view.mark_for_removal(id);
            }
        });
    }).wait();

    storage.free_marked_particles(); // Wywołanie bezargumentowe (dla wszystkich gatunków)

    EXPECT_EQ(chunk.count(), 3);

    auto result = read_pos_x(chunk);
    // Żywe cząstki to były: 0.0, 2.0, 4.0.
    // Indeks 5 jest martwy. Najbliższa żywa cząstka od końca to 4.0 (indeks 4).
    // Stąd 4.0 trafia na wolny slot 1.
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
}

/**
 * @test Verifies that free_marked_particles(species) affects only the requested species.
 */
TEST_F(ParticleStorageTest, SelectiveSpeciesRemoval) {
    ParticleStorage storage(q);
    SpeciesID electrons = static_cast<SpeciesID>(1);
    SpeciesID ions = static_cast<SpeciesID>(2);

    auto& chunk_e = storage.add_chunk(electrons, 10);
    populate_chunk(chunk_e, 3); // [0.0, 1.0, 2.0]

    auto& chunk_i = storage.add_chunk(ions, 10);
    populate_chunk(chunk_i, 3); // [0.0, 1.0, 2.0]

    // Oznaczamy po jednej cząstce w obu gatunkach
    auto view_e = chunk_e.get_view();
    auto view_i = chunk_i.get_view();

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            view_e.mark_for_removal(0);
            view_i.mark_for_removal(0);
        });
    }).wait();

    // Czyszczenie TYLKO dla elektronów
    storage.free_marked_particles(electrons);

    EXPECT_EQ(chunk_e.count(), 2); // Elektrony skompaktowane
    EXPECT_EQ(chunk_i.count(), 3); // Jony nieusunięte (oczekują w buforze)

    // Teraz czyszczenie jonów
    storage.free_marked_particles(ions);
    EXPECT_EQ(chunk_i.count(), 2);
}

/**
 * @test MultiChunkMultiSpeciesRemovalAndAllocationCheck
 * @brief Verifies particle removal, compaction, and chunk management parametrically, 
 *        safely capped based on available device VRAM.
 */
TEST_F(ParticleStorageTest, CompactionReleasesEmptyChunk)
{
   
}