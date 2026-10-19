#pragma once

#include <cmath>

#include "ISimulationStep.hpp"
#include "ParticleInjectorStrategy/IParticleInjectorStrategy.hpp"
#include "SimulationUtils.hpp"

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
    double number_density_;                                    ///< Physical particle number density \f$n\f$ [\f$\text{m}^{-3}\f$].
    double reference_volume_or_area_scale_;                    ///< Geometry scale parameter / volume substitute.
    std::unique_ptr<IParticleInjectorStrategy> strategy_;     ///< Geometry and generation strategy (Point/Beam).

public:
    /// @brief Constructs a particle injector step with a given strategy and physical parameters.
    ParticleInjectorStep(SpeciesID species, double density, double volume_or_area, 
                         std::unique_ptr<IParticleInjectorStrategy> strategy);

    /// @brief Executes the simulation step asynchronously using SYCL event dependencies.
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override;

    /// @brief Retrieves the human-readable name of the simulation step.
    std::string name() const override;

    /// @brief Computes the number of macroparticles to inject for the current time step.
    size_t compute_macro_particles_to_inject(const SimulationContext &ctx) const;
};