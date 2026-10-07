#pragma once

#include "ISimulationStep.hpp"

/**
 * @brief Simulation step that solves Maxwell's equations (or Poisson's equation) on the spatial grid.
 *
 * Computes updated electric (\f$\mathbf{E}\f$) and magnetic (\f$\mathbf{B}\f$) fields based on
 * current charge and current densities deposited on the grid.
 */
class FieldSolverStep : public ISimulationStep
{
public:
    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;
};
