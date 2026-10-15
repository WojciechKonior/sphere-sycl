#pragma once

#include "IParticleInjectorStrategy.hpp"

/**
 * @brief Beam, stream, or wall injection source strategy.
 * 
 * Emits particles over a circular disk normal to a specified velocity vector. 
 * A very large circle radius acts effectively as a full wall source.
 */
class BeamSourceStrategy : public IParticleInjectorStrategy {
public:
    BeamSourceStrategy(Vector3 center, Vector3 velocity, float circle_radius, float temperature) {}
};