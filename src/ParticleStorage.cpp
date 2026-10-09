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
            views.push_back({c->get_view()});
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

size_t ParticleStorage::get_total_count() const
{
    size_t total = 0;
    for (const auto &[species, chunks] : species_chunks_)
    {
        for (const auto &c : chunks)
            total += c->count();
    }
    return total;
}

size_t ParticleStorage::get_total_capacity() const
{
    size_t total = 0;
    for (const auto &[species, chunks] : species_chunks_)
    {
        for (const auto &c : chunks)
            total += c->capacity();
    }
    return total;
}

size_t ParticleStorage::get_chunk_count() const
{
    size_t total = 0;
    for (const auto &[species, chunks] : species_chunks_)
    {
        total += chunks.size();
    }
    return total;
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

void ParticleStorage::compact_species_chunks(SpeciesID species)
{
    auto &chunks = species_chunks_[species];
    if (chunks.empty()) return;

    // 1. Najpierw wykonujemy lokalną kompaktację w każdym chunku (usuwamy dziury wewnątrz chunków)
    for (auto &chunk : chunks)
    {
        if (chunk->count() > 0)
        {
            compact_chunk(*chunk);
        }
    }

    // 2. GLOBALNE ZAPEŁNIANIE WOLNYCH MIEJSC (Two-pointer Swap-and-Pop)
    size_t target_chunk_idx = 0;              // Wskaźnik "od przodu": przetwarza chunki z wolnym miejscem
    size_t source_chunk_idx = chunks.size() - 1; // Wskaźnik "od końca": zabiera cząstki z ostatniego niepustego chunka

    while (target_chunk_idx < source_chunk_idx)
    {
        auto &target_chunk = *chunks[target_chunk_idx];

        // Ile wolnych miejsc ma obecny chunk docelowy?
        size_t free_slots = target_chunk.capacity() - target_chunk.count();

        // Jeśli obecny chunk jest pełny, przechodzimy do kolejnego od przodu
        if (free_slots == 0)
        {
            target_chunk_idx++;
            continue;
        }

        // Szukamy od końca pierwszego chunka, który ma jakiekolwiek żywe cząstki do zabrania
        while (source_chunk_idx > target_chunk_idx && chunks[source_chunk_idx]->count() == 0)
        {
            source_chunk_idx--;
        }

        // Jeśli wskaźniki się spotkały lub minęły, nie ma już skąd brać cząstek
        if (source_chunk_idx <= target_chunk_idx)
        {
            break;
        }

        auto &source_chunk = *chunks[source_chunk_idx];

        // Wyznaczamy ile cząstek przemieścimy w tej iteracji:
        // Jest to minimum z liczby wolnych miejsc w docelowym i liczby dostępnych cząstek w źródłowym
        size_t particles_to_move = std::min(free_slots, source_chunk.count());

        size_t target_offset = target_chunk.count();
        size_t source_offset = source_chunk.count() - particles_to_move;

        // 3. Kopiowanie bloku cząstek na GPU dla SoA
        q_.submit([&](sycl::handler &cgh) {
            auto t_px = target_chunk.d_posX_; auto t_py = target_chunk.d_posY_; auto t_pz = target_chunk.d_posZ_;
            auto t_vx = target_chunk.d_velX_; auto t_vy = target_chunk.d_velY_; auto t_vz = target_chunk.d_velZ_;

            auto s_px = source_chunk.d_posX_; auto s_py = source_chunk.d_posY_; auto s_pz = source_chunk.d_posZ_;
            auto s_vx = source_chunk.d_velX_; auto s_vy = source_chunk.d_velY_; auto s_vz = source_chunk.d_velZ_;

            cgh.parallel_for(sycl::range<1>(particles_to_move), [=](sycl::id<1> idx) {
                size_t t_i = target_offset + idx[0];
                size_t s_i = source_offset + idx[0];

                t_px[t_i] = s_px[s_i];
                t_py[t_i] = s_py[s_i];
                t_pz[t_i] = s_pz[s_i];

                t_vx[t_i] = s_vx[s_i];
                t_vy[t_i] = s_vy[s_i];
                t_vz[t_i] = s_vz[s_i];
            });
        }).wait();

        // Zwiększamy licznik w chunku docelowym i zmniejszamy w źródłowym
        target_chunk.set_count(target_chunk.count() + particles_to_move);
        source_chunk.set_count(source_chunk.count() - particles_to_move);

        // Jeśli chunk źródłowy został opróżniony do 0, przesuwamy wskaźnik źródłowy w lewo
        if (source_chunk.count() == 0)
        {
            source_chunk_idx--;
        }
    }

    // 4. USUWAMY PUSTE CHUNKI Z WEKTORA ZWALNIAJĄC VRAM
    std::erase_if(chunks, [](const std::unique_ptr<Chunk> &chunk) {
        return chunk->count() == 0;
    });
}

void ParticleStorage::free_marked_particles(SpeciesID species)
{
    std::lock_guard<std::mutex> lock(storage_mutex_);
    compact_species_chunks(species);
}

void ParticleStorage::free_marked_particles()
{
    std::lock_guard<std::mutex> lock(storage_mutex_);
    for (auto &[species, chunks] : species_chunks_)
    {
        compact_species_chunks(species);
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