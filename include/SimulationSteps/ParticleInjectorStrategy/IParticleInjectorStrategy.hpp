#pragma once

/**
 * @brief Simple 3D vector structure for spatial positions and velocity directions.
 */
struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

/**
 * @brief Abstract interface for particle injection geometry and velocity distribution strategies.
 */
class IParticleInjectorStrategy{
public:
    virtual ~IParticleInjectorStrategy() = default;
    virtual double get_surface_area() const = 0;
    virtual double get_characteristic_velocity() const = 0;
};