#include "SimulationSteps/ParticleInjectorStep.hpp"

ParticleInjectorStep::ParticleInjectorStep(SpeciesID species, float density, float volume)
    : species_id_(species), number_density_(density), volume_(volume) {}

sycl::event ParticleInjectorStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    const auto &traits = ctx.species_registry.get(species_id_);

    double total_physical_particles = number_density_ * volume_;
    size_t macro_particles_to_inject = static_cast<size_t>(total_physical_particles / traits.weight);

    if (macro_particles_to_inject == 0)
    {
        return dependency;
    }

    auto handles = ctx.storage.allocate_space(species_id_, macro_particles_to_inject);

    std::vector<sycl::event> injection_events;
    injection_events.reserve(handles.size());

    for (auto &handle : handles)
    {
        size_t count = handle.count;
        size_t offset = handle.offset;

        // Wskaźniki przesunięte o offset odpowiadający początkowi nowego segmentu:
        float *d_px = handle.px + offset;
        float *d_py = handle.py + offset;
        float *d_pz = handle.pz + offset;
        float *d_vx = handle.vx + offset;
        float *d_vy = handle.vy + offset;
        float *d_vz = handle.vz + offset;

        sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh)
                                           {
                cgh.depends_on(dependency);

                cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    d_px[i] = 0.0f;
                    d_py[i] = 0.0f;
                    d_pz[i] = 0.0f;
                    d_vx[i] = 0.0f;
                    d_vy[i] = 0.0f;
                    d_vz[i] = 0.0f;
                }); });

        injection_events.push_back(evt);
    }

    std::cout << "  [" << name() << "] Dodano " << macro_particles_to_inject
              << " makrocząstek gatunku " << traits.name
              << " (waga w = " << traits.weight << ")\n";

    return ctx.queue.submit([&](sycl::handler &cgh)
                            {
            cgh.depends_on(injection_events);
            // cgh.single_task([=]() {});
            cgh.host_task([]() {}); });
}

std::string ParticleInjectorStep::name() const { return "Particle Injector"; }