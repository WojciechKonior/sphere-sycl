#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include "Species.hpp"

class ParticleStorage;
class GridField;

namespace SimulationUtils {
    /// @brief Physical and simulation configuration constants.
    namespace Constants {
        constexpr double MACROPARTICLE_WEIGHT    = 1.0e8f;
    
        constexpr double MASS_ELECTRON           = 9.109e-31f;
        constexpr double MASS_HYDROGEN_ION       = 1.672e-27f;
        constexpr double MASS_HYDROGEN           = 1.674e-27f;
    
        constexpr double CHARGE_ELEMENTARY       = 1.602e-19f;
        constexpr double CHARGE_ELECTRON         = -CHARGE_ELEMENTARY;
        constexpr double CHARGE_HYDROGEN_ION     =  CHARGE_ELEMENTARY;
        constexpr double CHARGE_NEUTRAL          =  0.0f;
        constexpr double k_B                     = 1.380649e-23; // Boltzmann constant
        constexpr double PI                      = 3.14159265358979323846;
    
        constexpr size_t CHUNK_CAPACITY         = 2000000; // 2e6 -> ((6*4 + 4/8)*2e6 + 8 + 8 + 4)B = 46,73MB per chunk
    }

    /// @brief Generates a high-quality pseudorandom float in [0, 1) using PCG Hash.
    /// @param seed Unique global identifier (e.g., global particle index).
    /// @param salt Salt or dimension discriminator (e.g., 1 for radius, 2 for theta, etc.).
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

    /// @brief Shared runtime state and resource references passed across all simulation steps.
    struct SimulationContext {
        sycl::queue &queue;                 ///< SYCL queue used for kernel submissions and memory operations
        ParticleStorage &storage;           ///< Container managing particle species and VRAM allocations
        GridField &grid;                    ///< Field solver data and spatial grid structures
        SpeciesRegistry &species_registry;  ///< Registry holding physical and numerical traits of active species
        
        double dt;                           ///< Time increment per simulation step [s]
        double current_time = 0.0f;          ///< Elapsed physical simulation time [s]
        size_t current_step = 0;            ///< Current timestep iteration index
    };
    
    /// @brief Simple 3D vector structure for spatial positions and velocity directions.
    struct Vector3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };
}

using namespace SimulationUtils;
