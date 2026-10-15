#include "SimulationSteps/ParticleInjectorStep.hpp"

ParticleInjectorStep::ParticleInjectorStep(SpeciesID species, double density, double volume_or_area, double temperature, 
                         std::unique_ptr<IParticleInjectorStrategy> strategy)
    : species_id_(species), 
      number_density_(density), 
      reference_volume_or_area_scale_(volume_or_area), 
      temperature_(temperature), 
      strategy_(std::move(strategy)) {}


size_t ParticleInjectorStep::compute_macro_particles_to_inject(const SimulationContext &ctx) const {
    const auto &traits = ctx.species_registry.get(species_id_);
    double dt = static_cast<double>(ctx.dt);

    double surface_area = static_cast<double>(strategy_->get_surface_area());
    double characteristic_velocity = static_cast<double>(strategy_->get_characteristic_velocity());

    double total_physical_particles = number_density_ * characteristic_velocity * surface_area * dt;
    double macro_particles_to_inject = total_physical_particles / static_cast<double>(traits.weight);

    size_t rounded_count = std::lround(macro_particles_to_inject);

    return rounded_count;
}

sycl::event ParticleInjectorStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    const auto &traits = ctx.species_registry.get(species_id_);
    // float dt = ctx.dt;

    // float surface_area = strategy_->get_surface_area();
    // float characteristic_velocity = strategy_->get_characteristic_velocity();

    // // 1. Obliczenie liczby makrocząstek do wstrzyknięcia
    // double total_physical_particles = number_density_ * characteristic_velocity * surface_area * dt;
    // size_t macro_particles_to_inject = static_cast<size_t>(total_physical_particles / traits.weight);
    size_t macro_particles_to_inject = compute_macro_particles_to_inject(ctx);

    if (macro_particles_to_inject == 0)
    {
        return dependency;
    }

    // 2. Rezerwacja slotów w pre-alokowanych chunkach VRAM (zwiększa liczniki w chunkach)
    auto handles = ctx.storage.reserve_space(species_id_, macro_particles_to_inject);

    std::vector<sycl::event> injection_events;
    injection_events.reserve(handles.size());

    size_t global_particle_offset = 0;
    float temperature = temperature_;
    float particle_mass = traits.elem_mass;

    // // 3. Iteracja po segmentach i uruchomienie równoległych kerneli na GPU
    // for (auto &handle : handles)
    // {
    //     size_t count = handle.count;
    //     size_t offset = handle.offset;

    //     // Bezpośrednie wskaźniki USM w pamięci urządzenia (GPU)
    //     float *d_px = handle.px + offset;
    //     float *d_py = handle.py + offset;
    //     float *d_pz = handle.pz + offset;
    //     float *d_vx = handle.vx + offset;
    //     float *d_vy = handle.vy + offset;
    //     float *d_vz = handle.vz + offset;

    //     uint64_t segment_global_offset = global_particle_offset;
    //     global_particle_offset += count;

    //     // Pobieramy specyficzne parametry geometrii ze strategii (np. dla PointSource lub BeamSource)
    //     // Możemy przekazać je przez wartości do lambdy kernela SYCL, unikając narzutu wirtualnego wewnątrz GPU.
        
    //     // Przykład dla PointSource (w docelowej architekturze strategia może dostarczać obiekt funkktora urządzeniowego):
    //     sycl::event kernel_evt = ctx.queue.submit([&](sycl::handler &cgh)
    //     {
    //         cgh.depends_on(dependency);

    //         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
    //             size_t i = idx[0];
    //             uint64_t seed_idx = segment_global_offset + i;

    //             // --- CAŁA LOGIKA GENEROWANIA WYKONUJE SIĘ RÓWNOLEGLE NA GPU ---
    //             // (Przykładowa implementacja inline dla źródła punktowego / termicznego)
                
    //             // Pozycja (np. domyślnie środek 0,0,0 lub z uwzględnieniem sfery)
    //             d_px[i] = 0.0f;
    //             d_py[i] = 0.0f;
    //             d_pz[i] = 0.0f;

    //             // Prędkość bazowa + składowa termiczna (uproszczony generator pseudolosowy na GPU)
    //             float vx_base = 1000.0f; 
    //             float vy_base = 0.0f;
    //             float vz_base = 0.0f;

    //             if (temperature > 0.0f) {
    //                 constexpr float k_B = 1.380649e-23f;
    //                 float v_th = sycl::sqrt(k_B * temperature / particle_mass);
                    
    //                 // Prosty deterministyczny szum oparty o seed indeksu
    //                 float pseudo_rand_x = static_cast<float>((seed_idx * 7) % 100 - 50) / 50.0f;
    //                 float pseudo_rand_y = static_cast<float>((seed_idx * 13) % 100 - 50) / 50.0f;
    //                 float pseudo_rand_z = static_cast<float>((seed_idx * 19) % 100 - 50) / 50.0f;

    //                 vx_base += pseudo_rand_x * v_th;
    //                 vy_base += pseudo_rand_y * v_th;
    //                 vz_base += pseudo_rand_z * v_th;
    //             }

    //             d_vx[i] = vx_base;
    //             d_vy[i] = vy_base;
    //             d_vz[i] = vz_base;
    //         });
    //     });

    //     injection_events.push_back(kernel_evt);
    // }

    // std::cout << "  [" << name() << "] Wstrzyknięto współbieżnie na GPU " << macro_particles_to_inject
    //           << " makrocząstek gatunku " << traits.name << " (dt = " << dt << " s)\n";

    return ctx.queue.submit([&](sycl::handler &cgh)
    {
        cgh.depends_on(injection_events);
        cgh.host_task([]() {});
    });
}

std::string ParticleInjectorStep::name() const { return "Particle Injector"; }