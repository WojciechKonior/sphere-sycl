#pragma once

#include <sycl/sycl.hpp>
#include <new>
#include "Species.hpp"

///@brief Lightweight Structure of Arrays (SoA) view used to pass device VRAM pointers into SYCL kernels.
struct ChunkSoAView {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t count;
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
    
    /// @return Remaining capacity available for new particles.
    size_t available_space() const;

    /// @return ChunkSoAView struct
    ChunkSoAView get_view() const;
    
private:    
    /// @brief Deallocates USM memory associated with this chunk.
    void release(sycl::queue &q);
};