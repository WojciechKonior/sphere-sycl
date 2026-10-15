#pragma once

#include "IParticleInjectorStrategy.hpp"

/**
 * @brief Point or spherical injection source strategy.
 * 
 * Generates particles around a central point, optionally distributed within a sphere 
 * of a given radius with thermal velocity spreading.
 */
class PointSourceStrategy : public IParticleInjectorStrategy {
public:
    PointSourceStrategy(Vector3 center, float mean_velocity, float sphere_radius) {}
};