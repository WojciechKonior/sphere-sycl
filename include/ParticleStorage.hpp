#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstddef>
#include <mutex>

#include "Species.hpp"
#include "Chunk.hpp"

/// @brief Lightweight handle passed to Injectors specifying the target VRAM region for writing new particles.
struct SegmentHandle {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t offset;
    size_t count;
};

/// @brief Manages VRAM allocations, chunk lifecycles, and particle memory (SoA layout) per species.
class ParticleStorage {
private:
    sycl::queue &q_;
    std::unordered_map<SpeciesID, std::vector<std::unique_ptr<Chunk>>> species_chunks_;
    mutable std::mutex storage_mutex_;

    /// @brief Internal function for Swap-and-Pop on a single chunk
    void compact_chunk(Chunk& chunk);

    /// @brief Internal function for Swap-and-Pop on a single species
    void compact_species_chunks(SpeciesID species);

public:
    explicit ParticleStorage(sycl::queue &q);
    ~ParticleStorage() = default;

    // Disable copying due to VRAM resource ownership
    ParticleStorage(const ParticleStorage&) = delete;
    ParticleStorage& operator=(const ParticleStorage&) = delete;

    ParticleStorage(ParticleStorage&&) = delete;
    ParticleStorage& operator=(ParticleStorage&&) = delete;

    /// @brief Allocates capacity for `count_to_add` particles, expanding VRAM chunks as needed.
    /// @return A list of segment handles providing writable device memory views for Injectors.
    std::vector<SegmentHandle> allocate_space(SpeciesID species, size_t count_to_add);

    /// @brief Retrieves SoA views for all active chunks of a specific species (used by Push/Scatter solvers).
    std::vector<ChunkSoAView> get_views(SpeciesID species);

    /// @brief Returns the total number of active particles for a given species.
    size_t get_total_count(SpeciesID species) const;

    /// @brief Returns the total allocated VRAM capacity (in particle units) for a given species.
    size_t get_total_capacity(SpeciesID species) const;

    /// @brief Returns the current number of allocated chunks for a given species.
    size_t get_chunk_count(SpeciesID species) const;

    /// @brief Returns the total number of active particles across all species.
    size_t get_total_count() const;

    /// @brief Returns the total allocated VRAM capacity (in particle units) across all species.
    size_t get_total_capacity() const;

    /// @brief Returns the total number of allocated chunks across all species.
    size_t get_chunk_count() const;

    /// @brief removing all marked particles for all species
    void free_marked_particles();

    /// @brief removing marked particles only for one species
    void free_marked_particles(SpeciesID species);

    /// @brief adding a chunk
    Chunk& add_chunk(SpeciesID species, size_t capacity);
};