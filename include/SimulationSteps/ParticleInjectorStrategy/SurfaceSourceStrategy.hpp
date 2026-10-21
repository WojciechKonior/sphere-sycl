#pragma once

#include "SimulationSteps/ParticleInjectorStrategy/IParticleInjectorStrategy.hpp"
#include "SimulationUtils.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/VelocityDistributionPolicy.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/FlatQuadeShape.hpp"

/**
 * @class SurfaceSourceStrategy
 * @brief Surface injection source strategy utilizing a generic shape policy (e.g., planar quadrilateral).
 * 
 * Generates particles within a thin surface volume layer defined by geometry boundaries
 * and flow velocity over a timestep dt, incorporating thermal velocity spreading.
 * 
 * @tparam ShapePolicy Geometry policy defining the surface shape (e.g., FlatQuadShape).
 * @tparam VelocityDistributionPolicy Policy governing thermal spreading (e.g., MaxwellBoltzmannDistribution).
 */
template <typename ShapePolicy, typename VelocityDistributionPolicy = MaxwellBoltzmannDistribution>
class SurfaceSourceStrategy : public IParticleInjectorStrategy {
private:
    ShapePolicy shape_;
    Vector3 velocity_;
    VelocityDistributionPolicy velocity_distribution_;

public:
    /**
     * @brief Constructs a SurfaceSourceStrategy.
     * 
     * @param shape Pre-configured shape instance (e.g., FlatQuadShape).
     * @param velocity 3D flow velocity vector [m/s].
     * @param dist_policy Velocity distribution policy object.
     */
    SurfaceSourceStrategy(FlatQuadShape shape, Vector3 velocity, VelocityDistributionPolicy dist_policy = {})
        : shape_(shape), velocity_(velocity), velocity_distribution_(dist_policy) {
            shape_.orient_towards_velocity(velocity_);
        }

    /**
     * @brief Computes the total surface area of the injection boundary.
     * @return Surface area [m^2].
     */
    double get_surface_area() const override {
        return shape_.get_area();
    }

    /**
     * @brief Returns the characteristic velocity magnitude of the source.
     * @return Speed magnitude [m/s].
     */
    double get_characteristic_velocity() const override {
        // Zwracamy wielkość (moduł) wektora prędkości do obliczeń strumienia makrocząstek
        return std::sqrt(velocity_.x * velocity_.x + 
                         velocity_.y * velocity_.y + 
                         velocity_.z * velocity_.z);
    }

    /**
     * @brief Computes the number of macro-particles to inject for the current time step.
     * 
     * Evaluates the physical particle flux through the surface: N = n * v_mean * area * dt,
     * and scales it by the species macro-particle weight.
     * 
     * @param ctx Shared runtime simulation context.
     * @param species_id Identifier of the particle species being injected.
     * @param number_density Target physical number density [m^-3].
     * @return Number of macro-particles to create in the current step.
     */
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

    /**
     * @brief Submits the SYCL kernel to initialize injected particle positions and velocities on the device.
     * 
     * @param handle Segment handle pointing to particle storage buffers for the current chunk.
     * @param params Global injection parameters (timestamps, masses, global offsets, etc.).
     * @param cgh SYCL command group handler for kernel submission.
     */
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
        auto param = params;

        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            uint64_t seed_idx = global_offset + i;

            Vector3 effective_normal = shape_copy.device_get_inward_normal();
            float normal_vel = static_cast<float>(velocity.x * effective_normal.x + 
                                                  velocity.y * effective_normal.y + 
                                                  velocity.z * effective_normal.z);
            float thickness = normal_vel * static_cast<float>(param.dt);

            // 1. Generowanie pozycji cząstki wewnątrz warstwy powierzchniowej (GPU-safe)
            Vector3 pos = shape_copy.device_generate_position(seed_idx, thickness, effective_normal);

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
     * @brief Generates injection velocity based on the 3D mean velocity vector with thermal spreading on the device.
     * 
     * @param seed_idx Unique seed index per particle.
     * @param velocity 3D flow velocity vector [m/s].
     * @param particle_mass Mass of a single physical particle [kg].
     * @param position 3D Cartesian position of the generated particle (unused in uniform flow).
     * @param shape Shape policy reference (unused).
     * @param distribution_policy Velocity distribution policy instance.
     * @return Generated 3D velocity vector [m/s].
     */
    static inline Vector3 device_generate_velocity(
        uint64_t seed_idx, 
        Vector3 velocity, 
        double particle_mass, 
        Vector3, 
        const ShapePolicy & , 
        VelocityDistributionPolicy distribution_policy
    ) {
        float vx_base = static_cast<float>(velocity.x);
        float vy_base = static_cast<float>(velocity.y);
        float vz_base = static_cast<float>(velocity.z);

        // Aplikujemy politykę rozkładu prędkości termicznych (np. Maxwell-Boltzmann)
        distribution_policy.apply(seed_idx, vx_base, vy_base, vz_base, particle_mass);

        return {vx_base, vy_base, vz_base};
    }
};