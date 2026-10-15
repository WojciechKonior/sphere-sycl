#pragma once

#include "ISimulationStep.hpp"
#include "ParticleInjectorStrategy/IParticleInjectorStrategy.hpp"

/**
 * @brief Simulation step responsible for injecting new macroparticles into the VRAM storage.
 *
 * Calculates the required number of macroparticles based on physical density and target volume,
 * allocates memory segments in ParticleStorage, and initializes new particle states on GPU.
 */
class ParticleInjectorStep : public ISimulationStep
{
private:
    SpeciesID species_id_;                                    ///< Target particle species identifier.
    float number_density_;                                    ///< Physical particle number density \f$n\f$ [\f$\text{m}^{-3}\f$].
    float reference_volume_or_area_scale_;                    ///< Geometry scale parameter / volume substitute.
    float temperature_;                                       ///< Temperature \f$T\f$ [K] for thermal velocity spreading.
    std::unique_ptr<IParticleInjectorStrategy> strategy_;     ///< Geometry and generation strategy (Point/Beam).

public:
    /// @brief Constructs a particle injector step with a given strategy and physical parameters.
    ParticleInjectorStep(SpeciesID species, float density, float volume_or_area, float temperature, 
                         std::unique_ptr<IParticleInjectorStrategy> strategy);

    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;
};