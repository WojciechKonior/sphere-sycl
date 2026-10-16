#pragma once

/**
 * @brief Simple 3D vector structure for spatial positions and velocity directions.
 */
struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

/// @brief Parametry fizyczne i środowiskowe przekazywane do strategii generowania na GPU.
struct InjectionParameters {
    double dt;
    double temperature;
    double particle_mass;
    uint64_t global_offset;
    size_t total_macro_particles;
};

/**
 * @brief Abstract interface for particle injection geometry and velocity distribution strategies.
 */
class IParticleInjectorStrategy{
public:
    virtual ~IParticleInjectorStrategy() = default;
    virtual double get_surface_area() const = 0;
    virtual double get_characteristic_velocity() const = 0;

    /// @brief Computes the number of macroparticles to inject for the current time step.
    virtual size_t compute_macro_particles_to_inject(const SimulationContext &ctx, SpeciesID species_id, double number_density) const = 0;

    /// @brief Deleguje generowanie cząstek na GPU używając skompaktowanych struktur.
    virtual void device_initialize_particles(const SegmentHandle &handle, const InjectionParameters &params, sycl::handler &cgh) const = 0;

};