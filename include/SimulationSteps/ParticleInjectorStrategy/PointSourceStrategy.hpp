#pragma once

#include "IParticleInjectorStrategy.hpp"

namespace SimulationUtils {
    /**
     * @brief Generates a high-quality pseudorandom float in [0, 1) using PCG Hash.
     * @param seed Unique global identifier (e.g., global particle index).
     * @param salt Salt or dimension discriminator (e.g., 1 for radius, 2 for theta, etc.).
     */
    inline float pcg_random_float(uint64_t seed, uint32_t salt)
    {
        // 1. PCG Hash core
        uint32_t combined = static_cast<uint32_t>(seed ^ (seed >> 32)) + salt * 0x9e3779b9u;
        uint32_t state = combined * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        uint32_t hash_val = (word >> 22u) ^ word;
    
        // 2. Convert to float [0, 1) with full mantissa precision
        return static_cast<float>(hash_val) / 4294967296.0f;
    }
}

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
        if (sphere_radius_ > 0.0) return 4.0 * Constants::PI * sphere_radius_ * sphere_radius_;
        return 0.0; 
    }

    double get_characteristic_velocity() const override {
        return mean_velocity_;
    }

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
        double temperature = params.temperature;
        double particle_mass = params.particle_mass;

        double mean_radial_velocity = mean_velocity_;
        Vector3 center = center_;
        double R1 = sphere_radius_;
        double dr = mean_radial_velocity * params.dt;
        double R2 = R1 + dr; // Promień sfery granicznej

        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            uint64_t seed_idx = global_offset + i;

            Vector3 pos = device_generate_position(seed_idx, R1, R2, center);

            d_px[i] = pos.x;
            d_py[i] = pos.y;
            d_pz[i] = pos.z;

            Vector3 vel = device_generate_velocity(seed_idx, mean_radial_velocity, temperature, particle_mass, pos, center);

            d_vx[i] = vel.x;
            d_vy[i] = vel.y;
            d_vz[i] = vel.z;
        });
    }

    static inline Vector3 device_generate_position(uint64_t seed_idx, double R1, double R2, Vector3 center) {
        // 1. Generowanie liczb losowych za pomocą PCG Hash
        float xi1 = SimulationUtils::pcg_random_float(seed_idx, 1); // Do promienia
        float xi2 = SimulationUtils::pcg_random_float(seed_idx, 2); // Do cos(theta)
        float xi3 = SimulationUtils::pcg_random_float(seed_idx, 3); // Do phi

        // 2. Modelowanie gęstości malejącej od R1 do R2
        // float r1 = static_cast<float>(R1);
        // float r2 = static_cast<float>(R2);
        // float r = r2 - (r2 - r1) * sycl::sqrt(xi1);

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

    static inline Vector3 device_generate_velocity(uint64_t seed_idx, double mean_radial_velocity, double temperature, double particle_mass, Vector3 position, Vector3 center) {
        // 4. Kierunek prędkości (radialnie na zewnątrz od środka sfery)
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

        // 5. Dodanie składowej termicznej (fizyczny rozkład Maxwella-Boltzmanna / Gaussa)
        if (temperature > 0.0 && particle_mass > 0.0) {
            // Odchylenie standardowe dla jednej składowej przestrzennej v_th = sqrt(k_B * T / m)
            float v_th = static_cast<float>(sycl::sqrt(Constants::k_B * temperature / particle_mass));

            // Pobieramy liczby losowe dla Boxa-Mullera (używamy sol od 4 do 7)
            float u1 = SimulationUtils::pcg_random_float(seed_idx, 4);
            float u2 = SimulationUtils::pcg_random_float(seed_idx, 5);
            float u3 = SimulationUtils::pcg_random_float(seed_idx, 6);
            float u4 = SimulationUtils::pcg_random_float(seed_idx, 7);

            // Zabezpieczenie przed log(0)
            u1 = sycl::fmax(u1, 1e-7f);
            u3 = sycl::fmax(u3, 1e-7f);

            // Transformacja Boxa-Mullera dla osi X i Y
            float mag = sycl::sqrt(-2.0f * sycl::log(u1));
            float z0 = mag * sycl::cos(2.0f * static_cast<float>(Constants::PI) * u2);
            float z1 = mag * sycl::sin(2.0f * static_cast<float>(Constants::PI) * u3); // u3 lub kolejna para dla Z

            // Dla osi Z bierzemy drugą parę (lub upraszczamy, ale dla pełnej niezależności dymensyjnej bierzemy osobną parę)
            float mag_z = sycl::sqrt(-2.0f * sycl::log(u3));
            float z2 = mag_z * sycl::cos(2.0f * static_cast<float>(Constants::PI) * u4);

            vx_base += z0 * v_th;
            vy_base += z1 * v_th;
            vz_base += z2 * v_th;
        }

        return {vx_base, vy_base, vz_base};
    }
};