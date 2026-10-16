#pragma once

#include "IParticleInjectorStrategy.hpp"
#include <cmath>

/**
 * @brief Beam, stream, or wall injection source strategy.
 * 
 * Emits particles over a circular disk normal to a specified velocity vector. 
 * A very large circle radius acts effectively as a full wall source.
 */
template <typename VelocityDistributionPolicy = MaxwellBoltzmannDistribution>
class BeamSourceStrategy : public IParticleInjectorStrategy {
private:
    Vector3 center_;
    Vector3 velocity_;
    double circle_radius_;
    double speed_mag_;
    VelocityDistributionPolicy velocity_distribution_;

public:
    BeamSourceStrategy(Vector3 center, Vector3 velocity, double circle_radius, double temperature, VelocityDistributionPolicy dist_policy = {})
        : center_(center), velocity_(velocity), circle_radius_(circle_radius), velocity_distribution_(dist_policy) {
        
        speed_mag_ = std::sqrt(velocity_.x * velocity_.x + velocity_.y * velocity_.y + velocity_.z * velocity_.z);
    }

    double get_surface_area() const override {
        if (circle_radius_ > 0.0) return Constants::PI * circle_radius_ * circle_radius_;
        return 0.0;
    }

    double get_characteristic_velocity() const override {
        return speed_mag_ > 0.0 ? speed_mag_ : 1.0;
    }

    size_t compute_macro_particles_to_inject(const SimulationContext &ctx, SpeciesID species_id, double number_density) const override {
        const auto &traits = ctx.species_registry.get(species_id);
        double dt = static_cast<double>(ctx.dt);
        double v_mean = static_cast<double>(get_characteristic_velocity());
        double surface_area = static_cast<double>(get_surface_area());

        double total_physical_particles = 0.0;

        if (surface_area > 0.0) {
            total_physical_particles = number_density * v_mean * surface_area * dt;
        } else {
            double distance = v_mean * dt;
            total_physical_particles = number_density * distance;
        }

        if (total_physical_particles <= 0.0) {
            return 0;
        }

        double macro_particles_to_inject = total_physical_particles / static_cast<double>(traits.weight);
        return static_cast<size_t>(std::lround(macro_particles_to_inject));
    }

    void device_initialize_particles(const SegmentHandle &handle, const InjectionParameters &params, sycl::handler &cgh) const override {}

};