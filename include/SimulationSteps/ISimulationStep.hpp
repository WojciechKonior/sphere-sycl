#pragma once

#include "SimulationContext.hpp"

/// @brief Abstract interface representing a single execution step in the PIC simulation pipeline.
class ISimulationStep {
public:
    virtual ~ISimulationStep() = default;

    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    virtual sycl::event execute(SimulationContext &ctx, sycl::event dependency) = 0;

    /// @brief Retrieves the human-readable name of the simulation step.
    virtual std::string name() const = 0;
};