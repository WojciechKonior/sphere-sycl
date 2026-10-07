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
                               chunk->count_, // offset wewnątrz chunka
                               alloc_size});

            chunk->count_ += alloc_size;
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

        new_chunk->count_ = alloc_size;
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
        if (c->count_ > 0)
        {
            views.push_back({c->d_posX_, c->d_posY_, c->d_posZ_,
                             c->d_velX_, c->d_velY_, c->d_velZ_,
                             c->count_});
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