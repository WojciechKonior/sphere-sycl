#include "SimulationSteps/ChargeDepositionStep.hpp"

sycl::event ChargeDepositionStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    // Wywołujemy zdefiniowane w GridField reset_charge_density(q)
    ctx.grid.reset_charge_density(ctx.queue);

    std::vector<sycl::event> deposition_events;

    for (SpeciesID id : ctx.species_registry.get_all_ids())
    {
        const auto &traits = ctx.species_registry.get(id);
        float q_macro = traits.macro_charge();

        auto views = ctx.storage.get_views(id);
        for (const auto &view : views)
        {
            if (view.count == 0)
                continue;

            ChunkSoAView local_view = view;

            sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh)
                                               {
                    cgh.depends_on(dependency);

                    cgh.parallel_for(sycl::range<1>(local_view.count), [local_view, q_macro](sycl::id<1> idx) {
                        size_t i = idx[0];
                        // Dostęp: local_view.px[i], local_view.py[i], local_view.pz[i]
                    }); });

            deposition_events.push_back(evt);
        }

        std::cout << "  [" << name() << "] Zdeponowano ładunek q_macro = "
                  << q_macro << " C dla " << traits.name << "\n";
    }

    if (deposition_events.empty())
    {
        return dependency;
    }

    return ctx.queue.submit([&](sycl::handler &cgh)
                            {
            cgh.depends_on(deposition_events);
            // cgh.single_task([=]() {});
            cgh.host_task([]() {}); });
}

std::string ChargeDepositionStep::name() const { return "Charge Deposition (CIC)"; }