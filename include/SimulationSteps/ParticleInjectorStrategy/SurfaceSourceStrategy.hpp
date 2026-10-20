#pragma once

#include "SimulationSteps/ParticleInjectorStrategy/IParticleInjectorStrategy.hpp"
#include "SimulationUtils.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/VelocityDistributionPolicy.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/FlatQuadeShape.hpp"

/**
 * @brief Surface injection source strategy utilizing a planar quadrilateral shape.
 * 
 * Generates particles within a thin flat quad volume layer defined by surface geometry
 * and flow velocity over a timestep dt, with thermal velocity spreading.
 */
template <typename ShapePolicy, typename VelocityDistributionPolicy = MaxwellBoltzmannDistribution>
class SurfaceSourceStrategy : public IParticleInjectorStrategy {
private:
    ShapePolicy shape_;
    Vector3 velocity_;
    VelocityDistributionPolicy velocity_distribution_;

public:
    /**
     * @brief Constructor for SurfaceSourceStrategy.
     * @param shape Pre-configured FlatQuadShape instance.
     * @param velocity Magnitude of the mean injection velocity [m/s].
     * @param dist_policy Velocity distribution policy (e.g., MaxwellBoltzmannDistribution).
     */
    SurfaceSourceStrategy(FlatQuadShape shape, Vector3 velocity, VelocityDistributionPolicy dist_policy = {})
        : shape_(shape), velocity_(velocity), velocity_distribution_(dist_policy) {}

    double get_surface_area() const override {
        return shape_.get_area();
    }

    double get_characteristic_velocity() const override {
        // Zwracamy wielkość (moduł) wektora prędkości do obliczeń strumienia makrocząstek
        return std::sqrt(velocity_.x * velocity_.x + 
                         velocity_.y * velocity_.y + 
                         velocity_.z * velocity_.z);
    }

    size_t compute_macro_particles_to_inject(const SimulationContext &ctx, SpeciesID species_id, double number_density) const override {
        const auto &traits = ctx.species_registry.get(species_id);
        double dt = static_cast<double>(ctx.dt);
        double v_mean = static_cast<double>(get_characteristic_velocity());
        double surface_area = static_cast<double>(get_surface_area());

        // Strumień fizycznych cząstek przez powierzchnię w czasie dt: N = n * v * A * dt
        double total_physical_particles = number_density * v_mean * surface_area * dt;

        double macro_particles_to_inject = total_physical_particles / static_cast<double>(traits.weight);
        return static_cast<size_t>(std::lround(macro_particles_to_inject));
    }

    void device_initialize_particles(const SegmentHandle &handle, const InjectionParameters &params, sycl::handler &cgh) const override {
        // Wyciągamy wskaźniki uwzględniając lokalny offset chunku
        float *d_px = handle.px + handle.offset;
        float *d_py = handle.py + handle.offset;
        float *d_pz = handle.pz + handle.offset;
        float *d_vx = handle.vx + handle.offset;
        float *d_vy = handle.vy + handle.offset;
        float *d_vz = handle.vz + handle.offset;
        
        size_t count = handle.count;
        uint64_t global_offset = params.global_offset;
        double particle_mass = params.particle_mass;

        // Kopiujemy potrzebne obiekty i zmienne do lambdy SYCL (bezpieczne dla urządzenia)
        auto shape_copy = shape_;
        Vector3 velocity = velocity_;
        auto distribution_policy = velocity_distribution_;

        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            uint64_t seed_idx = global_offset + i;

            // 1. Generowanie pozycji cząstki wewnątrz warstwy powierzchniowej (GPU-safe)
            Vector3 pos = shape_copy.device_generate_position(seed_idx);

            d_px[i] = pos.x;
            d_py[i] = pos.y;
            d_pz[i] = pos.z;

            // 2. Generowanie prędkości cząstki w oparciu o wektor prędkości i rozkład termiczny
            Vector3 vel = device_generate_velocity(seed_idx, velocity, particle_mass, pos, shape_copy, distribution_policy);

            d_vx[i] = vel.x;
            d_vy[i] = vel.y;
            d_vz[i] = vel.z;
        });
    }

private:
    /**
     * @brief Generates injection velocity based on the 3D mean velocity vector with thermal spreading.
     */
    static inline Vector3 device_generate_velocity(
        uint64_t seed_idx, 
        Vector3 mean_velocity, 
        double particle_mass, 
        Vector3 /*position*/, 
        const ShapePolicy & /*shape*/, 
        VelocityDistributionPolicy distribution_policy
    ) {
        float vx_base = static_cast<float>(mean_velocity.x);
        float vy_base = static_cast<float>(mean_velocity.y);
        float vz_base = static_cast<float>(mean_velocity.z);

        // Aplikujemy politykę rozkładu prędkości termicznych (np. Maxwell-Boltzmann)
        distribution_policy.apply(seed_idx, vx_base, vy_base, vz_base, particle_mass);

        return {vx_base, vy_base, vz_base};
    }
};