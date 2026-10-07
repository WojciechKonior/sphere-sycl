#include "SimulationSteps/ParticlePusherStep.hpp"

sycl::event ParticlePusherStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    std::vector<sycl::event> pusher_events;

    for (SpeciesID id : ctx.species_registry.get_all_ids())
    {
        const auto &traits = ctx.species_registry.get(id);
        float q_over_m = traits.q_over_m();

        auto views = ctx.storage.get_views(id);
        for (const auto &view : views)
        {
            if (view.count == 0)
                continue;

            float dt = ctx.dt;
            ChunkSoAView local_view = view;

            sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh)
                                               {
                    cgh.depends_on(dependency);

                    cgh.parallel_for(sycl::range<1>(local_view.count), [local_view, dt, q_over_m](sycl::id<1> idx) {
                        size_t i = idx[0];
                        // Aktualizacja prędkości i pozycji wewnątrz local_view.vx[i], local_view.px[i] itp.
                    }); });

            pusher_events.push_back(evt);
        }

        std::cout << "  [" << name() << "] Przesunięto cząstki " << traits.name
                  << " (q/m = " << q_over_m << " C/kg)\n";
    }

    if (pusher_events.empty())
    {
        return dependency;
    }

    return ctx.queue.submit([&](sycl::handler &cgh)
                            {
            cgh.depends_on(pusher_events);
            // cgh.single_task([=]() {});
            cgh.host_task([]() {}); });
}

std::string ParticlePusherStep::name() const { return "Particle Pusher (Boris)"; }