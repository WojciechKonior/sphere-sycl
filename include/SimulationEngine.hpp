#pragma once

#include <sycl/sycl.hpp>
#include <vector>

#include "ParticleStorage.hpp"
#include "GridField.hpp"
#include "Species.hpp"
#include "SimulationSteps/ISimulationStep.hpp"

/**
 * @brief Orchestrates and manages the execution pipeline of the simulation.
 *
 * The SimulationEngine holds the shared simulation state and a sequence of
 * simulation steps (e.g., injectors, pushers, field solvers) executed sequentially 
 * in each iteration of the main time loop.
 */
class SimulationEngine
{
private:
    SimulationContext ctx_; /// Shared execution context containing SYCL queues, grids, and particle storage.
    std::vector<std::unique_ptr<ISimulationStep>> steps_; /// Pipeline of simulation steps executed in order during each timestep.

public:
    SimulationEngine(sycl::queue &q, ParticleStorage &storage, GridField &grid, SpeciesRegistry &reg, float dt);

    /// @brief Adds a pre-constructed step to the simulation execution pipeline.
    void add_step(std::unique_ptr<ISimulationStep> step);

    /// @brief Constructs and adds a new simulation step directly into the pipeline.
    template <typename T, typename... Args>
    void add_step(Args &&...args)
    {
        steps_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    /// @brief Executes the simulation loop for a specified number of timesteps.
    void run(size_t total_steps);

    /// @brief Provides access to the internal simulation context.
    SimulationContext &context();
};

