#pragma once

#include "ISimulationStep.hpp"
#include "ParticleStorage.hpp"
#include "PusherStrategy/BorisPusherStrategy.hpp" // lub odpowiedni nagłówek strategii

/**
 * @class ParticlePusherStep
 * @brief Simulation step that advances particle positions and velocities using configured force policies (e.g., Boris Pusher).
 * 
 * Iterates through all particle species and their memory chunks, submitting GPU kernels
 * via SYCL to integrate particle trajectories.
 */
template <typename PusherPolicy>
class ParticlePusherStep : public ISimulationStep {
private:
    PusherPolicy pusher_policy_; ///< Strategy instance governing particle motion and force integration

public:
    /**
     * @brief Constructs a ParticlePusherStep with an optional pusher policy instance.
     * @param policy Policy object (defaults to PusherPolicy()).
     */
    // explicit ParticlePusherStep(PusherPolicy policy = PusherPolicy()) : pusher_policy_(policy) {}
    ParticlePusherStep() : pusher_policy_(PusherPolicy()) {}

    /**
     * @brief Executes the simulation step asynchronously using SYCL event dependencies.
     * 
     * @param ctx Shared runtime simulation context containing storage, grid, queues, and dt.
     * @param dependency SYCL event dependency from previous pipeline steps.
     * @return SYCL event representing the completion of all push operations.
     */
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override 
    {
        std::vector<sycl::event> pusher_events;

        // 1. Iterujemy po wszystkich zarejestrowanych gatunkach cząstek
        for (SpeciesID id : ctx.species_registry.get_all_ids())
        {
            const auto &traits = ctx.species_registry.get(id);
            float q_over_m = traits.q_over_m();

            // 2. Pobieramy widoki chunków (SoA) dla danego gatunku
            auto views = ctx.storage.get_views(id);
            for (const auto &view : views)
            {
                if (view.count == 0)
                    continue;

                float dt = ctx.dt;
                ChunkSoAView local_view = view;

                // 3. Submisja kernela SYCL dla konkretnego chunku
                sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh)
                {
                    // Każdy podzbiór musi zaczekać na zakończenie poprzedniego kroku symulacji
                    cgh.depends_on(dependency);

                    // Delegujemy fizykę do strategii (np. BorisPusherStrategy)
                    pusher_policy_.push(local_view, dt, q_over_m, ctx.grid, cgh);
                });

                pusher_events.push_back(evt);
            }

            std::cout << "  [" << name() << "] Przesunięto cząstki " << traits.name
                      << " (q/m = " << q_over_m << " C/kg)\n";
        }

        // Jeśli nie było żadnych cząstek do pchania, zwracamy wejściową zależność
        if (pusher_events.empty())
        {
            return dependency;
        }

        // 4. Agregacja wszystkich zdarzeń równoległych chunków w jedno zdarzenie końcowe
        sycl::event pusher_done = ctx.queue.submit([&](sycl::handler &cgh)
        {
            cgh.depends_on(pusher_events);
            // Pusta barierka synchronizacyjna spinająca wszystkie eventy chunków w jeden strumień wyjściowy
            cgh.host_task([]() {}); 
        });
        
        for (SpeciesID id : ctx.species_registry.get_all_ids())
        {            
            ctx.storage.free_marked_particles(id); // (zależnie od tego, jak przyjmuje dependencje)
        }

        std::cout << "  [" << name() << "] Zakończono pchanie i czyszczenie cząstek.\n";

        return pusher_done;
    }

    /**
     * @brief Retrieves the human-readable name of the simulation step.
     * @return Step identifier string.
     */
    std::string name() const override { return "Particle Pusher (Boris)"; }
};