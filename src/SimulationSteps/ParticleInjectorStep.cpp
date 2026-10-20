#include "SimulationSteps/ParticleInjectorStep.hpp"

ParticleInjectorStep::ParticleInjectorStep(SpeciesID species, double density, 
                         std::unique_ptr<IParticleInjectorStrategy> strategy)
    : species_id_(species), 
      number_density_(density), 
    //   reference_volume_or_area_scale_(volume_or_area),  
      strategy_(std::move(strategy)) {}


size_t ParticleInjectorStep::compute_macro_particles_to_inject(const SimulationContext &ctx) const {
    return strategy_->compute_macro_particles_to_inject(ctx, species_id_, number_density_);
}

sycl::event ParticleInjectorStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    double dt = static_cast<double>(ctx.dt);
    const auto &traits = ctx.species_registry.get(species_id_);
    size_t macro_particles_to_inject = compute_macro_particles_to_inject(ctx);
    if (macro_particles_to_inject == 0){
        return dependency;
    }

    // 2. Rezerwacja slotów w pre-alokowanych chunkach VRAM (zwiększa liczniki w chunkach)
    auto handles = ctx.storage.reserve_space(species_id_, macro_particles_to_inject);

    std::vector<sycl::event> injection_events;
    injection_events.reserve(handles.size());

    size_t global_particle_offset = 0;

    // Przygotowanie wspólnych parametrów fizycznych
    InjectionParameters params{
        .dt = dt,
        .particle_mass = traits.elem_mass * traits.weight,
        .global_offset = 0 // Będziemy aktualizować w pętli
    };

    // 3. Iteracja po segmentach i uruchomienie równoległych kerneli na GPU
    for (auto &handle : handles)
    {
        // Aktualizujemy globalny offset dla bieżącego segmentu
        params.global_offset = global_particle_offset;
        global_particle_offset += handle.count;

        sycl::event kernel_evt = ctx.queue.submit([&](sycl::handler &cgh)
        {
            cgh.depends_on(dependency);
            strategy_->device_initialize_particles(handle, params, cgh);
        });

        injection_events.push_back(kernel_evt);
    }

    return injection_events.empty() ? dependency : injection_events.back();
}

std::string ParticleInjectorStep::name() const { return "Particle Injector"; }