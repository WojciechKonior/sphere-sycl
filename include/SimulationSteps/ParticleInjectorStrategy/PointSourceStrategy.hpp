#pragma once

#include "IParticleInjectorStrategy.hpp"
#include "SimulationUtils.hpp"
#include "VelocityDistributionPolicy.hpp"

/**
 * @brief Point or spherical injection source strategy.
 * 
 * Generates particles around a central point, optionally distributed within a sphere 
 * of a given radius with thermal velocity spreading.
 */
template <typename VelocityDistributionPolicy = MaxwellBoltzmannDistribution>
class PointSourceStrategy : public IParticleInjectorStrategy {
private:
    Vector3 center_;
    double mean_velocity_;
    double sphere_radius_;
    VelocityDistributionPolicy velocity_distribution_;

public:
    /**
     * @brief Constructs a PointSourceStrategy.
     * 
     * @param center Central coordinates of the injection source.
     * @param mean_velocity Mean radial velocity of injected particles [m/s].
     * @param sphere_radius Inner radius of the shell (R1). If 0.0, acts as a volumetric point source.
     * @param dist_policy Velocity distribution policy object.
     */
    PointSourceStrategy(Vector3 center, double mean_velocity, double sphere_radius, VelocityDistributionPolicy dist_policy = {})
        : center_(center), mean_velocity_(mean_velocity), sphere_radius_(sphere_radius), velocity_distribution_(dist_policy) {}

    /// @brief Computes the surface area of the injection boundary.
    double get_surface_area() const override {
        if (sphere_radius_ > 0.0) return 4.0 * Constants::PI * sphere_radius_ * sphere_radius_;
        return 0.0; 
    }

    /// @brief Returns the characteristic velocity of the source.
    double get_characteristic_velocity() const override {
        return mean_velocity_;
    }

    /**
     * @brief Calculates the number of macro-particles to inject in the current timestep.
     * 
     * Handles two distinct cases:
     * - Case 1 (Surface/Shell, R1 > 0): Flux-based injection through the sphere surface.
     * - Case 2 (Volumetric point source, R1 = 0): Volume-filling injection inside the sphere of radius R2 = v_mean * dt.
     * 
     * @param ctx Shared runtime simulation context.
     * @param species_id Identifier of the particle species being injected.
     * @param number_density Target physical number density [m^-3].
     * @return Number of macro-particles to create.
     */
    size_t compute_macro_particles_to_inject(const SimulationContext &ctx, SpeciesID species_id, double number_density) const override {
        const auto &traits = ctx.species_registry.get(species_id);
        double dt = static_cast<double>(ctx.dt);
        double v_mean = static_cast<double>(get_characteristic_velocity());
        double surface_area = static_cast<double>(get_surface_area());

        double total_physical_particles = 0.0;

        // Sprawdzamy, czy strategia to PointSourceStrategy z R1 == 0 (źródło objętościowe w kuli R2)
        // (Można to też wyciągnąć jako wirtualną metodę np. strategy->compute_total_physical_particles(ctx))
        if (surface_area > 0.0) {
            // Przypadek 1: Klasyczny strumień przez powierzchnię (dla powłoki R1 > 0 lub BeamSource)
            total_physical_particles = number_density * v_mean * surface_area * dt;
        } else {
            // Przypadek 2: Źródło punktowe (R1 = 0) - wypełnianie kuli o promieniu R2 = v_mean * dt
            double R2 = v_mean * dt;
            double volume_sphere = (4.0 / 3.0) * Constants::PI * R2 * R2 * R2;
            total_physical_particles = number_density * volume_sphere;
        }

        double macro_particles_to_inject = total_physical_particles / static_cast<double>(traits.weight);
        return static_cast<size_t>(std::lround(macro_particles_to_inject)); // lub std::lround
    }

    /**
     * @brief Submits the SYCL kernel to initialize injected particle positions and velocities on the device.
     * 
     * @param handle Segment handle pointing to particle storage buffers for the current chunk.
     * @param params Injection parameters including global offsets, masses, and timestamps.
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

        double mean_radial_velocity = mean_velocity_;
        Vector3 center = center_;
        double R1 = sphere_radius_;
        double dr = mean_radial_velocity * params.dt;
        double R2 = R1 + dr; // Promień sfery granicznej

        auto distribution_policy = velocity_distribution_;

        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            uint64_t seed_idx = global_offset + i;

            Vector3 pos = device_generate_position(seed_idx, R1, R2, center);

            d_px[i] = pos.x;
            d_py[i] = pos.y;
            d_pz[i] = pos.z;

            Vector3 vel = device_generate_velocity(seed_idx, mean_radial_velocity, particle_mass, pos, center, distribution_policy);

            d_vx[i] = vel.x;
            d_vy[i] = vel.y;
            d_vz[i] = vel.z;
        });
    }

    /**
     * @brief Generates a random position on the device within a spherical shell or solid sphere.
     * 
     * @param seed_idx Unique seed index per particle (derived from global particle index).
     * @param R1 Inner radius of the shell [m].
     * @param R2 Outer radius of the shell or sphere [m].
     * @param center Central point coordinates of the source.
     * @return Generated 3D Cartesian position vector.
     */
    static inline Vector3 device_generate_position(uint64_t seed_idx, double R1, double R2, Vector3 center) {
        // 1. Generowanie liczb losowych za pomocą PCG Hash
        float xi1 = SimulationUtils::pcg_random_float(seed_idx, 1); // Do promienia
        float xi2 = SimulationUtils::pcg_random_float(seed_idx, 2); // Do cos(theta)
        float xi3 = SimulationUtils::pcg_random_float(seed_idx, 3); // Do phi

        float r = 0.0f;
        
        // 2. Rozróżnienie strategii gęstości w zależności od R1
        if (R1 > 0.0) {
            // Przypadek 1: Powłoka sferyczna [R1, R2] z gęstością malejącą od R1 do R2
            float r1 = static_cast<float>(R1);
            float r2 = static_cast<float>(R2);
            r = r2 - (r2 - r1) * sycl::sqrt(xi1);
        } else {
            // Przypadek 2: Źródło czysto punktowe (R1 = 0) - równomierny rozkład objętościowy w kuli [0, R2]
            // Używamy pierwiastka trzeciego stopnia (cbrt) dla zachowania jednorodności w 3D (dV ~ r^2 dr)
            float r2 = static_cast<float>(R2);
            r = r2 * sycl::cbrt(xi1);
        }

        // 3. Losowe kąty na sferze
        float cos_theta = 1.0f - 2.0f * xi2; // [-1, 1]
        float sin_theta = sycl::sqrt(1.0f - cos_theta * cos_theta);
        float phi = 2.0f * static_cast<float>(Constants::PI) * xi3; // [0, 2*pi]

        // 4. Pozycja kartezjańska cząstki
        float px = static_cast<float>(center.x) + r * sin_theta * sycl::cos(phi);
        float py = static_cast<float>(center.y) + r * sin_theta * sycl::sin(phi);
        float pz = static_cast<float>(center.z) + r * cos_theta;

        return {px, py, pz};
    }
    
    /**
     * @brief Generates initial velocity vector with thermal spreading on the device.
     * 
     * @param seed_idx Unique seed index per particle.
     * @param mean_radial_velocity Mean radial drift velocity magnitude [m/s].
     * @param particle_mass Mass of a single physical particle [kg].
     * @param position 3D Cartesian position of the generated particle.
     * @param center Central point coordinates of the source.
     * @param distribution_policy Velocity distribution policy instance.
     * @return Generated 3D velocity vector [m/s].
     */
    static inline Vector3 device_generate_velocity(uint64_t seed_idx, double mean_radial_velocity, double particle_mass, Vector3 position, Vector3 center, VelocityDistributionPolicy distribution_policy) {
        float dx = position.x - static_cast<float>(center.x);
        float dy = position.y - static_cast<float>(center.y);
        float dz = position.z - static_cast<float>(center.z);
        float r = sycl::sqrt(dx * dx + dy * dy + dz * dz);

        float inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
        float nx = dx * inv_r;
        float ny = dy * inv_r;
        float nz = dz * inv_r;

        float v_mean = static_cast<float>(mean_radial_velocity);
        float vx_base = nx * v_mean;
        float vy_base = ny * v_mean;
        float vz_base = nz * v_mean;

        distribution_policy.apply(seed_idx, vx_base, vy_base, vz_base, particle_mass);

        return {vx_base, vy_base, vz_base};
    }
};