#pragma once

#include "IParticleInjectorStrategy.hpp"

/**
 * @brief Point or spherical injection source strategy.
 * 
 * Generates particles around a central point, optionally distributed within a sphere 
 * of a given radius with thermal velocity spreading.
 */
class PointSourceStrategy : public IParticleInjectorStrategy {
private:
    Vector3 center_;
    double mean_velocity_;
    double sphere_radius_;

public:
    PointSourceStrategy(Vector3 center, double mean_velocity, double sphere_radius)
        : center_(center), mean_velocity_(mean_velocity), sphere_radius_(sphere_radius) {}

    double get_surface_area() const override {
        // Dla źródła sferycznego powierzchnia emisyjna to 4 * pi * r^2
        // Jeśli promień jest zero (punkt czysto matematyczny), zwracamy 1.0 jako jednostkowy przelicznik skali
        if (sphere_radius_ > 0.0f) {
            constexpr double pi = 3.14159265f;
            return 4.0f * pi * sphere_radius_ * sphere_radius_;
        }
        return 1.0f; 
    }

    double get_characteristic_velocity() const override {
        return mean_velocity_;
    }
};