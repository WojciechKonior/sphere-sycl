#pragma once

#include "IParticleInjectorStrategy.hpp"
#include <cmath>

/**
 * @brief Beam, stream, or wall injection source strategy.
 * 
 * Emits particles over a circular disk normal to a specified velocity vector. 
 * A very large circle radius acts effectively as a full wall source.
 */
class BeamSourceStrategy : public IParticleInjectorStrategy {
private:
    Vector3 center_;
    Vector3 velocity_;
    double circle_radius_;
    double speed_mag_;

public:
    BeamSourceStrategy(Vector3 center, Vector3 velocity, double circle_radius, double temperature)
        : center_(center), velocity_(velocity), circle_radius_(circle_radius) {
        
        speed_mag_ = std::sqrt(velocity_.x * velocity_.x + velocity_.y * velocity_.y + velocity_.z * velocity_.z);
    }

    double get_surface_area() const override {
        // Powierzchnia dysku/tarczy/ściany: pi * r^2
        if (circle_radius_ > 0.0f) {
            constexpr double pi = 3.14159265f;
            return pi * circle_radius_ * circle_radius_;
        }
        return 1.0f;
    }

    double get_characteristic_velocity() const override {
        return speed_mag_ > 0.0f ? speed_mag_ : 1.0f;
    }
};