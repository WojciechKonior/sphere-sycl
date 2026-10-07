#pragma once

#include "ISimulationStep.hpp"

/**
 * @brief Simulation step responsible for injecting new macroparticles into the VRAM storage.
 *
 * Calculates the required number of macroparticles based on physical density and target volume,
 * allocates memory segments in ParticleStorage, and initializes new particle states on GPU.
 */
class ParticleInjectorStep : public ISimulationStep
{
private:
    SpeciesID species_id_;  /// Target particle species identifier to be injected.
    float number_density_;  /// Physical particle number density \f$n\f$ \f$[\text{m}^{-3}]\f$.
    float volume_;          /// Physical volume of the injection region \f$[\text{m}^3]\f$.

public:
    ParticleInjectorStep(SpeciesID species, float density, float volume);

    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;
};