#pragma once

#include "SimulationUtils.hpp"
#include "ParticleStorage.hpp"


/// @brief Parametry fizyczne i środowiskowe przekazywane do strategii generowania na GPU.
struct InjectionParameters {
    double dt;
    double particle_mass;
    uint64_t global_offset;
};

/**
 * @brief Abstract interface for particle injection geometry and velocity distribution strategies.
 */
class IParticleInjectorStrategy{
public:
    virtual ~IParticleInjectorStrategy() = default;

    /**
     * @brief Computes the total surface area of the injection boundary.
     * @return Surface area [m^2], or 0.0 for volumetric/point sources.
     */
    virtual double get_surface_area() const = 0;

    /**
     * @brief Returns the characteristic or mean velocity of the injected particles.
     * @return Characteristic velocity magnitude [m/s].
     */
    virtual double get_characteristic_velocity() const = 0;

    /**
     * @brief Computes the number of macro-particles to inject for the current time step.
     * 
     * @param ctx Shared runtime simulation context.
     * @param species_id Identifier of the particle species being injected.
     * @param number_density Target physical number density [m^-3].
     * @return Number of macro-particles to create in the current step.
     */
    virtual size_t compute_macro_particles_to_inject(const SimulationContext &ctx, SpeciesID species_id, double number_density) const = 0;

    /**
     * @brief Dispatches particle initialization on the device using compacted memory buffers.
     * 
     * Submits a SYCL parallel execution kernel to generate and initialize positions and velocities
     * for newly injected particles within the allocated memory segment chunk.
     * 
     * @param handle Segment handle pointing to particle storage buffers for the current chunk.
     * @param params Global injection parameters (timestamps, masses, global offsets, etc.).
     * @param cgh SYCL command group handler for kernel submission.
     */
    virtual void device_initialize_particles(const SegmentHandle &handle, const InjectionParameters &params, sycl::handler &cgh) const = 0;

};