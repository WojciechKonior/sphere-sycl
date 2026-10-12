#pragma once

#include <sycl/sycl.hpp>
#include <new>
#include "Species.hpp"

///@brief Lightweight Structure of Arrays (SoA) view used to pass device VRAM pointers into SYCL kernels.
struct ChunkSoAView {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t count;
    size_t capacity;
    int32_t* d_dead_indices;
    int32_t* d_dead_count;
    int8_t* d_is_dead;

    /// @brief Device method that submits a single-task in SYCL kernel to safely mark particle on GPU.
    inline void mark_for_removal_device(size_t local_idx) const {
        if (d_dead_count == nullptr || d_dead_indices == nullptr || local_idx >= count) {
            return;
        }

        // 2. Atomowe przydzielenie slotu
        auto atomic_dead_count = sycl::atomic_ref<
            int32_t, 
            sycl::memory_order::relaxed, 
            sycl::memory_scope::device, 
            sycl::access::address_space::global_space
        >(*d_dead_count);

        int32_t slot = atomic_dead_count.fetch_add(1);

        // 3. Zapis indeksu z ochroną przed OOB Write
        if (slot < static_cast<int32_t>(capacity)) {
            d_dead_indices[slot] = static_cast<int32_t>(local_idx);
        }
    }

// void ParticleStorage::make_is_dead_array(Chunk& chunk, uint8_t* d_is_dead) {
//     int32_t dead_count = 0;
//     q_.memcpy(&dead_count, chunk.d_dead_count_, sizeof(int32_t)).wait();

//     if (dead_count <= 0) return;

//     size_t capacity = chunk.capacity();
//     dead_count = std::min(dead_count, static_cast<int32_t>(chunk.count()));

//     // 1. Zerujemy maskę d_is_dead dla całego pojemnika
//     q_.memset(d_is_dead, 0, capacity * sizeof(uint8_t));

//     // 2. Wpisujemy 1 pod indeksy martwych cząstek (wielokrotny zapis tego samego indeksu = 1)
//     q_.submit([&](sycl::handler& cgh) {
//         auto* dead_indices = chunk.d_dead_indices_;

//         cgh.parallel_for(sycl::range<1>(dead_count), [=](sycl::id<1> idx) {
//             int32_t p_idx = dead_indices[idx];
//             if (p_idx >= 0 && static_cast<size_t>(p_idx) < capacity) {
//                 d_is_dead[p_idx] = 1;
//             }
//         });
//     }).wait();
// }

    /// @brief Host wrapper method that submits a single-task SYCL kernel to safely mark particle on GPU.
    sycl::event mark_for_removal(sycl::queue &q, size_t local_idx) const {
        if (d_dead_count == nullptr || d_dead_indices == nullptr || local_idx >= count)
            return sycl::event();

        auto view = *this;

        return q.submit([=](sycl::handler &cgh) {
            cgh.single_task([=]() {
                // Wywołanie dedykowanej funkcji Device!
                view.mark_for_removal_device(local_idx);
            });
        });
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