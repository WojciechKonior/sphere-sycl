#pragma once

#include <sycl/sycl.hpp>
#include <new>
#include "Species.hpp"

///@brief Lightweight Structure of Arrays (SoA) view used to pass device VRAM pointers into SYCL kernels.
struct ChunkSoAView {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t count;
    int32_t* d_dead_indices;
    int32_t* d_dead_count;

    /// @brief Safe method to call from SYCL kernel / host thread.
    void mark_for_removal(size_t local_idx) const {
        // Atomic refference to dead particle counter
        auto atomic_dead_count = sycl::atomic_ref<
            int32_t, 
            sycl::memory_order::relaxed, 
            sycl::memory_scope::device, 
            sycl::access::address_space::global_space
        >(*d_dead_count);

        // Unique slot in dead indices buffer
        int32_t slot = atomic_dead_count.fetch_add(1);
        d_dead_indices[slot] = static_cast<int32_t>(local_idx);
    }
};

/// @brief Manages a contiguous USM device allocation (Chunk) for particle data in SoA layout.
class Chunk {
    friend class ParticleStorage;

private:
    sycl::queue q_;
    size_t capacity_;
    size_t count_ = 0;

    float *d_posX_ = nullptr; float *d_posY_ = nullptr; float *d_posZ_ = nullptr;
    float *d_velX_ = nullptr; float *d_velY_ = nullptr; float *d_velZ_ = nullptr;

    int32_t *d_dead_indices_ = nullptr;
    int32_t *d_dead_count_ = nullptr;
    
public:
    Chunk(sycl::queue &q, size_t capacity = Constants::CHUNK_CAPACITY);
    ~Chunk();
    Chunk(const Chunk &) = delete; // Prevent copying to ensure sole ownership of VRAM resource allocations
    Chunk &operator=(const Chunk &) = delete; // Prevent copying to ensure sole ownership of VRAM resource allocations
    Chunk(Chunk &&other) noexcept; // Move constructor transfering VRAM ownership without reallocating
    Chunk &operator=(Chunk &&other) noexcept; 
    
    
    /// @return Maximum number of particles this chunk can hold.
    size_t capacity() const;
    
    /// @return Current active particle count.
    size_t count() const;

    /// @brief To set count_ 
    void set_count(size_t count);
    
    /// @return Remaining capacity available for new particles.
    size_t available_space() const;

    /// @return ChunkSoAView struct
    ChunkSoAView get_view() const;
    
private:    
    /// @brief Deallocates USM memory associated with this chunk.
    void release(sycl::queue &q);
};