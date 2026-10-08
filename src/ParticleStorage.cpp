#include "ParticleStorage.hpp"

ParticleStorage::ParticleStorage(sycl::queue &q) : q_(q) {}

std::vector<SegmentHandle> ParticleStorage::allocate_space(SpeciesID species, size_t count_to_add)
{
    std::vector<SegmentHandle> handles;
    if (count_to_add == 0)
        return handles;

    auto &chunks = species_chunks_[species];
    size_t remaining = count_to_add;

    // 1. Wypełnij dostępne wolne miejsca w istniejących chunkach
    for (auto &chunk : chunks)
    {
        if (remaining == 0)
            break;

        size_t space = chunk->available_space();
        if (space > 0)
        {
            size_t alloc_size = std::min(remaining, space);

            handles.push_back({chunk->d_posX_, chunk->d_posY_, chunk->d_posZ_,
                               chunk->d_velX_, chunk->d_velY_, chunk->d_velZ_,
                               chunk->count(), // offset wewnątrz chunka
                               alloc_size});

            chunk->set_count(chunk->count() + alloc_size);
            remaining -= alloc_size;
        }
    }

    // 2. Jeśli nadal brakuje miejsca, alokuj nowe PACZKI (Chunki) o stałej pojemności
    while (remaining > 0)
    {
        chunks.push_back(std::make_unique<Chunk>(q_, Constants::CHUNK_CAPACITY));
        auto &new_chunk = chunks.back();

        size_t alloc_size = std::min(remaining, new_chunk->capacity());

        handles.push_back({new_chunk->d_posX_, new_chunk->d_posY_, new_chunk->d_posZ_,
                           new_chunk->d_velX_, new_chunk->d_velY_, new_chunk->d_velZ_,
                           0, // nowy chunk zaczyna się od offsetu 0
                           alloc_size});

        new_chunk->set_count(alloc_size);
        remaining -= alloc_size;
    }

    return handles;
}

std::vector<ChunkSoAView> ParticleStorage::get_views(SpeciesID species)
{
    std::vector<ChunkSoAView> views;
    auto it = species_chunks_.find(species);
    if (it == species_chunks_.end())
        return views;

    for (auto &c : it->second)
    {
        if (c->count() > 0)
        {
            views.push_back({c->d_posX_, c->d_posY_, c->d_posZ_,
                             c->d_velX_, c->d_velY_, c->d_velZ_,
                             c->count()});
        }
    }
    return views;
}

size_t ParticleStorage::get_total_count(SpeciesID species) const
{
    auto it = species_chunks_.find(species);
    if (it == species_chunks_.end())
        return 0;

    size_t total = 0;
    for (const auto &c : it->second)
        total += c->count();
    return total;
}

size_t ParticleStorage::get_total_capacity(SpeciesID species) const
{
    auto it = species_chunks_.find(species);
    if (it == species_chunks_.end())
        return 0;

    size_t total = 0;
    for (const auto &c : it->second)
        total += c->capacity();
    return total;
}

size_t ParticleStorage::get_chunk_count(SpeciesID species) const
{
    auto it = species_chunks_.find(species);
    if (it == species_chunks_.end())
        return 0;
    return it->second.size();
}

void ParticleStorage::compact_chunk(Chunk& chunk) {
    int32_t dead_count = 0;

    // 1. Kopiujemy TYLKO 1 int (bardzo tani transfer), żeby sprawdzić czy są martwe cząstki
    q_.memcpy(&dead_count, chunk.d_dead_count_, sizeof(int32_t)).wait();

    if (dead_count == 0) return;

    dead_count = std::min(dead_count, static_cast<int32_t>(chunk.count()));
    size_t total_count = chunk.count();

    // 2. Alokujemy bufor na przesunięciatymczasowo na GPU (Stream Compaction / Swap-and-Pop na GPU)
    auto* d_moves = sycl::malloc_device<std::pair<int32_t, int32_t>>(dead_count, q_);
    auto* d_move_count = sycl::malloc_device<int32_t>(1, q_);
    q_.memset(d_move_count, 0, sizeof(int32_t)).wait();

    // 3. Kernel na GPU: Dopasowuje wolne sloty z lewej strony do żywych cząstek z prawej strony
    // (Odpowiednik Swap-and-Pop na GPU bez udziału CPU)
    q_.submit([&](sycl::handler& cgh) {
        auto* dead_indices = chunk.d_dead_indices_;

        cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            int32_t source_ptr = static_cast<int32_t>(total_count) - 1;

            for (int i = 0; i < dead_count; ++i) {
                int32_t target_slot = dead_indices[i];

                // Szukamy najbliższej żywej cząstki od końca chunka
                while (source_ptr > target_slot) {
                    bool is_dead = false;
                    for (int j = 0; j < dead_count; ++j) {
                        if (dead_indices[j] == source_ptr) {
                            is_dead = true;
                            break;
                        }
                    }
                    if (!is_dead) break;
                    source_ptr--;
                }

                if (source_ptr > target_slot) {
                    int32_t idx = d_move_count[0]++;
                    d_moves[idx] = {target_slot, source_ptr};
                    source_ptr--;
                } else {
                    break;
                }
            }
        });
    }).wait();

    // 4. Pobieramy liczbę faktycznych przesunięć
    int32_t num_moves = 0;
    q_.memcpy(&num_moves, d_move_count, sizeof(int32_t)).wait();

    // 5. Wykonujemy przemieszczenie danych na GPU dla pól SoA
    if (num_moves > 0) {
        q_.submit([&](sycl::handler& cgh) {
            auto posX = chunk.d_posX_; auto posY = chunk.d_posY_; auto posZ = chunk.d_posZ_;
            auto velX = chunk.d_velX_; auto velY = chunk.d_velY_; auto velZ = chunk.d_velZ_;

            cgh.parallel_for(sycl::range<1>(num_moves), [=](sycl::id<1> idx) {
                auto move = d_moves[idx];
                int32_t target = move.first;
                int32_t source = move.second;

                posX[target] = posX[source];
                posY[target] = posY[source];
                posZ[target] = posZ[source];

                velX[target] = velX[source];
                velY[target] = velY[source];
                velZ[target] = velZ[source];
            });
        }).wait();
    }

    // Sprzątamy bufor roboczy i resetujemy stan
    sycl::free(d_moves, q_);
    sycl::free(d_move_count, q_);

    chunk.set_count(chunk.count() - dead_count);
    q_.memset(chunk.d_dead_count_, 0, sizeof(int32_t)).wait();
}

void ParticleStorage::free_marked_particles(SpeciesID species)
{
    std::lock_guard<std::mutex> lock(storage_mutex_);

    auto it = species_chunks_.find(species);
    if (it == species_chunks_.end())
        return;

    for (auto &chunk : it->second)
    {
        if (chunk->count() > 0)
        {
            compact_chunk(*chunk);
        }
    }
}

void ParticleStorage::free_marked_particles()
{
    std::lock_guard<std::mutex> lock(storage_mutex_);

    for (auto &[species, chunks] : species_chunks_)
    {
        for (auto &chunk : chunks)
        {
            if (chunk->count() > 0)
            {
                compact_chunk(*chunk);
            }
        }
    }
}

Chunk &ParticleStorage::add_chunk(SpeciesID species, size_t capacity)
{
    std::lock_guard<std::mutex> lock(storage_mutex_);
    auto chunk = std::make_unique<Chunk>(q_, capacity);
    Chunk &ref = *chunk;
    species_chunks_[species].push_back(std::move(chunk));
    return ref;
}