#pragma once

#include "ISimulationStep.hpp"

/// @brief Simulation step that calculates stochastic particle collisions using Monte Carlo methods (MCC).
class ParticleColliderStep : public ISimulationStep
{
public:
    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;
};
