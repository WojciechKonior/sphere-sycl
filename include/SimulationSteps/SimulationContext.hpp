#pragma once

#include "ParticleStorage.hpp"
#include "GridField.hpp"
#include "Species.hpp"

/// @brief Shared runtime state and resource references passed across all simulation steps.
struct SimulationContext {
    sycl::queue &queue;                 ///< SYCL queue used for kernel submissions and memory operations
    ParticleStorage &storage;           ///< Container managing particle species and VRAM allocations
    GridField &grid;                    ///< Field solver data and spatial grid structures
    SpeciesRegistry &species_registry;  ///< Registry holding physical and numerical traits of active species
    
    float dt;                           ///< Time increment per simulation step [s]
    float current_time = 0.0f;          ///< Elapsed physical simulation time [s]
    size_t current_step = 0;            ///< Current timestep iteration index
};