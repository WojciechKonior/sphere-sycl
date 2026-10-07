#include "SimulationEngine.hpp"

SimulationEngine::SimulationEngine(sycl::queue &q, ParticleStorage &storage, GridField &grid, SpeciesRegistry &reg, float dt)
    : ctx_{q, storage, grid, reg, dt} {}

void SimulationEngine::add_step(std::unique_ptr<ISimulationStep> step)
{
    steps_.push_back(std::move(step));
}

void SimulationEngine::run(size_t total_steps)
{
    std::cout << "=========================================================\n";
    std::cout << " Uruchomienie SimulationEngine (" << steps_.size() << " kroków w potoku)\n";
    std::cout << "=========================================================\n";

    sycl::event last_event; // Domyślnie utworzony event oznaczający gotowość

    for (size_t step = 0; step < total_steps; ++step)
    {
        std::cout << "\n---> Krok " << ctx_.current_step << " | t = " << ctx_.current_time << " s <---\n";

        for (auto &step_ptr : steps_)
        {
            // Przepływ zdarzeń wewnątrz potoku kroków
            last_event = step_ptr->execute(ctx_, last_event);
        }

        // Opcjonalne oczekiwanie na zakończenie całej ramki czasowej
        last_event.wait();

        ctx_.current_time += ctx_.dt;
        ctx_.current_step++;
    }
}

SimulationContext &SimulationEngine::context() { return ctx_; }