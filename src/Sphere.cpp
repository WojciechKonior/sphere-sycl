#include "ParticleStorage.hpp"


// -----------------------------------------------------------------------------
// 2. Zaślepki dla Klas Pamięci (ParticleStorage) oraz Siatki (GridField)
// -----------------------------------------------------------------------------

class GridField {
public:
    explicit GridField(sycl::queue &q) {}

    void reset_charge_density(sycl::queue &q) {
        // Czyszczenie bufora gęstości ładunku rho na GPU
    }
};

// -----------------------------------------------------------------------------
// 3. Kontekst Symulacji
// -----------------------------------------------------------------------------

struct SimulationContext {
    sycl::queue &queue;
    ParticleStorage &storage;
    GridField &grid;
    SpeciesRegistry &species_registry;
    
    float dt;
    float current_time = 0.0f;
    size_t current_step = 0;
};

// -----------------------------------------------------------------------------
// 4. Interfejs Kroku Symulacji
// -----------------------------------------------------------------------------

class ISimulationStep {
public:
    virtual ~ISimulationStep() = default;
    virtual sycl::event execute(SimulationContext &ctx, sycl::event dependency) = 0;
    virtual std::string name() const = 0;
};

// -----------------------------------------------------------------------------
// 5. Konkretne Klasy Kroków Symulacji
// -----------------------------------------------------------------------------



class ParticleInjectorStep : public ISimulationStep {
private:
    SpeciesID species_id_;
    float number_density_; // Fizyczna gęstość n [m^-3]
    float volume_;         // Objętość obszaru wstrzykiwania [m^3]

public:
    ParticleInjectorStep(SpeciesID species, float density, float volume)
        : species_id_(species), number_density_(density), volume_(volume) {}

    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override {
        const auto &traits = ctx.species_registry.get(species_id_);
        
        double total_physical_particles = number_density_ * volume_;
        size_t macro_particles_to_inject = static_cast<size_t>(total_physical_particles / traits.weight);

        if (macro_particles_to_inject == 0) {
            return dependency;
        }

        auto handles = ctx.storage.allocate_space(species_id_, macro_particles_to_inject);

        std::vector<sycl::event> injection_events;
        injection_events.reserve(handles.size());

        for (auto &handle : handles) {
            size_t count = handle.count;
            size_t offset = handle.offset;
            
            // Wskaźniki przesunięte o offset odpowiadający początkowi nowego segmentu:
            float* d_px = handle.px + offset;
            float* d_py = handle.py + offset;
            float* d_pz = handle.pz + offset;
            float* d_vx = handle.vx + offset;
            float* d_vy = handle.vy + offset;
            float* d_vz = handle.vz + offset;

            sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dependency);

                cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    d_px[i] = 0.0f;
                    d_py[i] = 0.0f;
                    d_pz[i] = 0.0f;
                    d_vx[i] = 0.0f;
                    d_vy[i] = 0.0f;
                    d_vz[i] = 0.0f;
                });
            });

            injection_events.push_back(evt);
        }

        std::cout << "  [" << name() << "] Dodano " << macro_particles_to_inject 
                  << " makrocząstek gatunku " << traits.name 
                  << " (waga w = " << traits.weight << ")\n";

        return ctx.queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(injection_events);
            // cgh.single_task([=]() {});
            cgh.host_task([]() {});
        });
    }

    std::string name() const override { return "Particle Injector"; }
};

// Krok 2: Depozycja ładunku z cząstek na siatkę (Cloud-in-Cell)
class ChargeDepositionStep : public ISimulationStep {
public:
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override {
        // Wywołujemy zdefiniowane w GridField reset_charge_density(q)
        ctx.grid.reset_charge_density(ctx.queue);

        std::vector<sycl::event> deposition_events;

        for (SpeciesID id : ctx.species_registry.get_all_ids()) {
            const auto &traits = ctx.species_registry.get(id);
            float q_macro = traits.macro_charge();

            auto views = ctx.storage.get_views(id);
            for (const auto &view : views) {
                if (view.count == 0) continue;

                ChunkSoAView local_view = view;

                sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(dependency);

                    cgh.parallel_for(sycl::range<1>(local_view.count), [local_view, q_macro](sycl::id<1> idx) {
                        size_t i = idx[0];
                        // Dostęp: local_view.px[i], local_view.py[i], local_view.pz[i]
                    });
                });

                deposition_events.push_back(evt);
            }

            std::cout << "  [" << name() << "] Zdeponowano ładunek q_macro = " 
                      << q_macro << " C dla " << traits.name << "\n";
        }

        if (deposition_events.empty()) {
            return dependency;
        }

        return ctx.queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deposition_events);
            // cgh.single_task([=]() {});
            cgh.host_task([]() {});
        });
    }

    std::string name() const override { return "Charge Deposition (CIC)"; }
};

// Krok 3: Solwer Pól (Poisson / Yee)
class FieldSolverStep : public ISimulationStep {
public:
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override {
        sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependency);
            cgh.single_task([=]() {
                // Kod solwera na GPU
            });
        });

        std::cout << "  [" << name() << "] Obliczono pola E oraz B na siatce.\n";
        return evt;
    }

    std::string name() const override { return "Field Solver"; }
};

// Krok 4: Pychacz Cząstek (Boris Pusher)
class ParticlePusherStep : public ISimulationStep {
public:
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override {
        std::vector<sycl::event> pusher_events;

        for (SpeciesID id : ctx.species_registry.get_all_ids()) {
            const auto &traits = ctx.species_registry.get(id);
            float q_over_m = traits.q_over_m();

            auto views = ctx.storage.get_views(id);
            for (const auto &view : views) {
                if (view.count == 0) continue;

                float dt = ctx.dt;
                ChunkSoAView local_view = view;

                sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(dependency);

                    cgh.parallel_for(sycl::range<1>(local_view.count), [local_view, dt, q_over_m](sycl::id<1> idx) {
                        size_t i = idx[0];
                        // Aktualizacja prędkości i pozycji wewnątrz local_view.vx[i], local_view.px[i] itp.
                    });
                });

                pusher_events.push_back(evt);
            }

            std::cout << "  [" << name() << "] Przesunięto cząstki " << traits.name 
                      << " (q/m = " << q_over_m << " C/kg)\n";
        }

        if (pusher_events.empty()) {
            return dependency;
        }

        return ctx.queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(pusher_events);
            // cgh.single_task([=]() {});
            cgh.host_task([]() {});
        });
    }

    std::string name() const override { return "Particle Pusher (Boris)"; }
};

// Krok 5: Zderzenia cząstek (Monte Carlo Collisions - MCC)
class ParticleColliderStep : public ISimulationStep {
public:
    sycl::event execute(SimulationContext &ctx, sycl::event dependency) override {
        sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependency);
            cgh.single_task([=]() {
                // Kod MCC na GPU
            });
        });

        std::cout << "  [" << name() << "] Przeliczono losowe zderzenia cząstek.\n";
        return evt;
    }

    std::string name() const override { return "Particle Collider (MCC)"; }
};

// -----------------------------------------------------------------------------
// Silnik Symulacji (SimulationEngine)
// -----------------------------------------------------------------------------

class SimulationEngine {
private:
    SimulationContext ctx_;
    std::vector<std::unique_ptr<ISimulationStep>> steps_;

public:
    SimulationEngine(sycl::queue &q, ParticleStorage &storage, GridField &grid, SpeciesRegistry &reg, float dt)
        : ctx_{q, storage, grid, reg, dt} {}

    void add_step(std::unique_ptr<ISimulationStep> step) {
        steps_.push_back(std::move(step));
    }

    template <typename T, typename... Args>
    void add_step(Args&&... args) {
        steps_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    void run(size_t total_steps) {
        std::cout << "=========================================================\n";
        std::cout << " Uruchomienie SimulationEngine (" << steps_.size() << " kroków w potoku)\n";
        std::cout << "=========================================================\n";

        sycl::event last_event; // Domyślnie utworzony event oznaczający gotowość

        for (size_t step = 0; step < total_steps; ++step) {
            std::cout << "\n---> Krok " << ctx_.current_step << " | t = " << ctx_.current_time << " s <---\n";

            for (auto &step_ptr : steps_) {
                // Przepływ zdarzeń wewnątrz potoku kroków
                last_event = step_ptr->execute(ctx_, last_event);
            }

            // Opcjonalne oczekiwanie na zakończenie całej ramki czasowej
            last_event.wait();

            ctx_.current_time += ctx_.dt;
            ctx_.current_step++;
        }
    }

    SimulationContext& context() { return ctx_; }
};

// -----------------------------------------------------------------------------
// 7. Przykład Użycia (main)
// -----------------------------------------------------------------------------

int main() {
    sycl::queue q;

    // 1. Inicjalizacja składowych
    ParticleStorage storage(q);
    GridField grid(q);
    SpeciesRegistry species_reg;

    // 2. Rejestracja gatunków z wykorzystaniem przestrzeni Constants
    species_reg.register_species(
        SpeciesID::ELECTRON, 
        "Elektron", 
        Constants::MASS_ELECTRON, 
        Constants::CHARGE_ELECTRON, 
        Constants::MACROPARTICLE_WEIGHT
    );

    species_reg.register_species(
        SpeciesID::HYDROGEN_ION, 
        "Proton", 
        Constants::MASS_HYDROGEN_ION, 
        Constants::CHARGE_HYDROGEN_ION, 
        Constants::MACROPARTICLE_WEIGHT
    );

    // 3. Utworzenie silnika symulacji (dt = 1 ps)
    SimulationEngine engine(q, storage, grid, species_reg, 1.0e-12f);

    // 4. Budowanie potoku wykonywania (Pipeline)
    engine.add_step<ParticleInjectorStep>(SpeciesID::ELECTRON, 1.0e18f /* n */, 1.0e-6f /* V */);
    engine.add_step<ChargeDepositionStep>();
    engine.add_step<FieldSolverStep>();
    engine.add_step<ParticlePusherStep>();
    engine.add_step<ParticleColliderStep>();

    // 5. Uruchomienie symulacji
    engine.run(2);

    return 0;
}
