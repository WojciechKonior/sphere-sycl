#pragma once
#include <cstdint>


namespace SimulationUtils {
    /**
     * @brief Generates a high-quality pseudorandom float in [0, 1) using PCG Hash.
     * @param seed Unique global identifier (e.g., global particle index).
     * @param salt Salt or dimension discriminator (e.g., 1 for radius, 2 for theta, etc.).
     */
    inline float pcg_random_float(uint64_t seed, uint32_t salt)
    {
        // 1. PCG Hash core
        uint32_t combined = static_cast<uint32_t>(seed ^ (seed >> 32)) + salt * 0x9e3779b9u;
        uint32_t state = combined * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        uint32_t hash_val = (word >> 22u) ^ word;
    
        // 2. Convert to float [0, 1) with full mantissa precision
        return static_cast<float>(hash_val) / 4294967296.0f;
    }
}