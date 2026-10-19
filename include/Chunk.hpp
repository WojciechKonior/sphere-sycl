#pragma once

#include <sycl/sycl.hpp>
#include <new>
#include "Species.hpp"
#include "SimulationUtils.hpp"

///@brief Lightweight Structure of Arrays (SoA) view used to pass device VRAM pointers into SYCL kernels.
struct ChunkSoAView {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t count;
    size_t capacity;
    int32_t* d_dead_count;
    uint32_t* d_is_dead;

    /// @return true if partcle is marked as dead (for removal), false otherwise
    bool device_is_particle_dead(int32_t i) const
    {
        size_t word_idx = static_cast<size_t>(i) / 32;
        uint32_t bit_mask = 1u << (static_cast<size_t>(i) % 32);
        return (d_is_dead[word_idx] & bit_mask) != 0;
    }

    /// @brief Device method that submits a single-task in SYCL kernel to safely mark particle on GPU.
    inline void device_mark_for_removal(size_t local_idx) const {
        if (d_dead_count == nullptr || d_is_dead == nullptr || local_idx >= count) {
            return;
        }

        // 1. Obliczamy indeks słowa (uint32_t) oraz konkretny bit dla danej cząstki
        size_t word_idx = local_idx / 32;
        uint32_t bit_mask = 1u << (local_idx % 32);

        // 2. Atomowa operacja bitowa OR na słowie (zwraca poprzednią wartość słowa)
        auto atomic_word = sycl::atomic_ref<
            uint32_t, 
            sycl::memory_order::relaxed, 
            sycl::memory_scope::device, 
            sycl::access::address_space::global_space
        >(d_is_dead[word_idx]);

        uint32_t old_word = atomic_word.fetch_or(bit_mask);

        // 3. Sprawdzamy, czy dany bit był już ustawiony (duplikat)
        if ((old_word & bit_mask) != 0) {
            return; // Cząstka była już oznaczona – przerywamy, brak duplikatu w indices!
        }

        // 4. Pierwsze oznaczenie – bezpiecznie dodajemy indeks do d_dead_indices
        auto atomic_dead_count = sycl::atomic_ref<
            int32_t, 
            sycl::memory_order::relaxed, 
            sycl::memory_scope::device, 
            sycl::access::address_space::global_space
        >(*d_dead_count);

        int32_t slot = atomic_dead_count.fetch_add(1);
    }

    /// @brief Host wrapper method that submits a single-task SYCL kernel to safely mark particle on GPU.
    sycl::event mark_for_removal(sycl::queue &q, size_t local_idx) const {
        if (d_dead_count == nullptr || local_idx >= count)
            return sycl::event();

        auto view = *this;

        return q.submit([=](sycl::handler &cgh) {
            cgh.single_task([=]() {
                // Wywołanie dedykowanej funkcji Device!
                view.device_mark_for_removal(local_idx);
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

    int32_t *d_dead_count_ = nullptr;
    uint32_t *d_is_dead_ = nullptr;
    
public:
    explicit Chunk(sycl::queue &q, size_t capacity = Constants::CHUNK_CAPACITY);
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