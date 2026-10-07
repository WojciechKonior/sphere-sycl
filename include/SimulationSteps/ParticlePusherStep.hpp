#pragma once

#include "ISimulationStep.hpp"

/// @brief Simulation step that advances particle positions and velocities using field forces (Boris Pusher algorithm).
class ParticlePusherStep : public ISimulationStep
{
public:
    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;
};