#pragma once

#include "ISimulationStep.hpp"

/**
 * @brief Simulation step that deposits macroparticle charges onto the spatial grid.
 *
 * Uses weight-assignment schemes (e.g., Cloud-in-Cell / CIC) to interpolate continuous particle positions
 * onto discrete grid nodes, updating the scalar charge density field \f$\rho\mathbf{(x)}\f$.
 */
class ChargeDepositionStep : public ISimulationStep
{
public:
    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;
};
