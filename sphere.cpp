#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

namespace Constants {
    constexpr float MACROPARTICLE_WEIGHT = 1.0e8f;

    constexpr float MASS_ELECTRON = 9.109e-31f;
    constexpr float MASS_HYDROGEN_ION   = 1.672e-27f;
    constexpr float MASS_HYDROGEN = 1.674e-27f;

    constexpr float CHARGE_ELEMENTARY = 1.602e-19f;
    constexpr float CHARGE_ELECTRON   = -CHARGE_ELEMENTARY;
    constexpr float CHARGE_HYDROGEN_ION     =  CHARGE_ELEMENTARY;
    constexpr float CHARGE_NEUTRAL    =  0.0f;

    constexpr size_t CHUNK_CAPACITY = 10'000'000; // 1e7
}

enum class SpeciesID {
    ELECTRON,
    HYDROGEN_ION,
    NEUTRAL_HYDROGEN
};

// -----------------------------------------------------------------------------
// 1. Definicje Identyfikatorów i Rejestr Cech Gatunków
// -----------------------------------------------------------------------------
struct SpeciesTraits {
    SpeciesID id;
    std::string name;
    
    float elem_mass;    // Masa 1 cząstki fizycznej [kg]
    float elem_charge;  // Ładunek 1 cząstki fizycznej [C]
    float weight;       // Waga makrocząstki w (ile cząstek fizycznych zastępuje 1 makrocząstka)

    // Stałe przeliczane dla pojedynczej makrocząstki
    float macro_mass() const { return elem_mass * weight; }
    float macro_charge() const { return elem_charge * weight; }

    // Stosunek q/m (waga się skraca – taki sam dla cząstki fizycznej i makrocząstki)
    float q_over_m() const { return elem_charge / elem_mass; }
};

class SpeciesRegistry {
private:
    std::unordered_map<SpeciesID, SpeciesTraits> registry_;

public:
    void register_species(SpeciesID id, const std::string &name, float m, float q, float weight) {
        registry_[id] = SpeciesTraits{id, name, m, q, weight};
    }

    const SpeciesTraits& get(SpeciesID id) const {
        return registry_.at(id);
    }

    std::vector<SpeciesID> get_all_ids() const {
        std::vector<SpeciesID> ids;
        for (const auto &[id, traits] : registry_) {
            ids.push_back(id);
        }
        return ids;
    }
};






// Lekka struktura przekazywana do Injektorów: wskazuje dokładnie, gdzie zapisać nowe cząstki
struct SegmentHandle {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t offset;
    size_t count;
};

// Widok SoA przekazywany do Push / Scatter (Read-Only lub Mutator)
struct ChunkSoAView {
    float* px; float* py; float* pz;
    float* vx; float* vy; float* vz;
    size_t count;
};










class Chunk {
    friend class ParticleStorage;

private:
    size_t capacity_;
    size_t count_ = 0;

    // Surowe wskaźniki VRAM
    float *d_posX_ = nullptr; float *d_posY_ = nullptr; float *d_posZ_ = nullptr;
    float *d_velX_ = nullptr; float *d_velY_ = nullptr; float *d_velZ_ = nullptr;
    
public:
    Chunk(sycl::queue &q, size_t capacity = Constants::CHUNK_CAPACITY)
        : capacity_(capacity), count_(0) 
    {
        size_t bytes = capacity_ * sizeof(float);
        d_posX_ = static_cast<float *>(sycl::malloc_device(bytes, q));
        d_posY_ = static_cast<float *>(sycl::malloc_device(bytes, q));
        d_posZ_ = static_cast<float *>(sycl::malloc_device(bytes, q));
        d_velX_ = static_cast<float *>(sycl::malloc_device(bytes, q));
        d_velY_ = static_cast<float *>(sycl::malloc_device(bytes, q));
        d_velZ_ = static_cast<float *>(sycl::malloc_device(bytes, q));
    }

    void release(sycl::queue &q) {
        if (d_posX_) sycl::free(d_posX_, q);
        if (d_posY_) sycl::free(d_posY_, q);
        if (d_posZ_) sycl::free(d_posZ_, q);
        if (d_velX_) sycl::free(d_velX_, q);
        if (d_velY_) sycl::free(d_velY_, q);
        if (d_velZ_) sycl::free(d_velZ_, q);
        d_posX_ = d_posY_ = d_posZ_ = d_velX_ = d_velY_ = d_velZ_ = nullptr;
    }

    ~Chunk() = default;

    // Blokujemy kopiowanie dla bezpieczeństwa VRAM
    Chunk(const Chunk &) = delete;
    Chunk &operator=(const Chunk &) = delete;

    // Ruch (Move semantics) pozwala przechowywać Chunki w std::vector
    Chunk(Chunk &&other) noexcept
        : capacity_(other.capacity_), count_(other.count_),
          d_posX_(other.d_posX_), d_posY_(other.d_posY_), d_posZ_(other.d_posZ_),
          d_velX_(other.d_velX_), d_velY_(other.d_velY_), d_velZ_(other.d_velZ_) 
    {
        other.capacity_ = 0; other.count_ = 0;
        other.d_posX_ = other.d_posY_ = other.d_posZ_ = nullptr;
        other.d_velX_ = other.d_velY_ = other.d_velZ_ = nullptr;
    }

    size_t capacity() const { return capacity_; }
    size_t count() const { return count_; }
    size_t available_space() const { return capacity_ - count_; }
};













class ParticleStorage {
private:
    sycl::queue &q_;

    // Mapa przechowywania chunków per gatunek cząstek
    std::unordered_map<SpeciesID, std::vector<Chunk>> species_chunks_;

public:
    explicit ParticleStorage(sycl::queue &q) : q_(q) {}

    ~ParticleStorage() {
        // Czyszczenie całej pamięci VRAM we wszystkich chunkach
        for (auto &[species, chunks] : species_chunks_) {
            for (auto &chunk : chunks) {
                chunk.release(q_);
            }
        }
    }

    // --- ALOKACJA I ROZBUDOWA KONTENERA O PACZKI ---

    /// @brief Przydziela miejsce na `count_to_add` cząstek. W razie potrzeby tworzy nowe paczki (Chunki).
    /// @return Wektor segmentów (rękojeści), do których Injektor może wpisać dane.
    std::vector<SegmentHandle> allocate_space(SpeciesID species, size_t count_to_add) {
        std::vector<SegmentHandle> handles;
        if (count_to_add == 0) return handles;

        auto &chunks = species_chunks_[species];
        size_t remaining = count_to_add;

        // 1. Wypełnij dostępne wolne miejsca w istniejących chunkach
        for (auto &chunk : chunks) {
            if (remaining == 0) break;

            size_t space = chunk.available_space();
            if (space > 0) {
                size_t alloc_size = std::min(remaining, space);

                handles.push_back({
                    chunk.d_posX_, chunk.d_posY_, chunk.d_posZ_,
                    chunk.d_velX_, chunk.d_velY_, chunk.d_velZ_,
                    chunk.count_, // offset wewnątrz chunka
                    alloc_size
                });

                chunk.count_ += alloc_size;
                remaining -= alloc_size;
            }
        }

        // 2. Jeśli nadal brakuje miejsca, alokuj nowe PACZKI (Chunki) o stałej pojemności
        while (remaining > 0) {
            chunks.emplace_back(q_, Constants::CHUNK_CAPACITY);
            auto &new_chunk = chunks.back();

            size_t alloc_size = std::min(remaining, new_chunk.capacity());

            handles.push_back({
                new_chunk.d_posX_, new_chunk.d_posY_, new_chunk.d_posZ_,
                new_chunk.d_velX_, new_chunk.d_velY_, new_chunk.d_velZ_,
                0, // nowy chunk zaczyna się od offsetu 0
                alloc_size
            });

            new_chunk.count_ = alloc_size;
            remaining -= alloc_size;
        }

        return handles;
    }

    // --- INTERFEJSY ODCZYTU/I MIGRACJI DLA SOLVERÓW ---

    /// @brief Zwraca widok na wszystkie chunki danej gatunku cząstek (dla Pushera / Scattera)
    std::vector<ChunkSoAView> get_views(SpeciesID species) {
        std::vector<ChunkSoAView> views;
        auto it = species_chunks_.find(species);
        if (it == species_chunks_.end()) return views;

        for (auto &c : it->second) {
            if (c.count_ > 0) {
                views.push_back({
                    c.d_posX_, c.d_posY_, c.d_posZ_,
                    c.d_velX_, c.d_velY_, c.d_velZ_,
                    c.count_
                });
            }
        }
        return views;
    }

    /// @brief Sumaryczna liczba aktywnych cząstek danego gatunku
    size_t get_total_count(SpeciesID species) const {
        auto it = species_chunks_.find(species);
        if (it == species_chunks_.end()) return 0;

        size_t total = 0;
        for (const auto &c : it->second) total += c.count();
        return total;
    }

    /// @brief Całkowita przydzielona pojemność VRAM dla gatunku
    size_t get_total_capacity(SpeciesID species) const {
        auto it = species_chunks_.find(species);
        if (it == species_chunks_.end()) return 0;

        size_t total = 0;
        for (const auto &c : it->second) total += c.capacity();
        return total;
    }

    /// @brief Całkowita liczba chunków dla danego gatunku
    size_t get_chunk_count(SpeciesID species) const {
        auto it = species_chunks_.find(species);
        if (it == species_chunks_.end()) return 0;
        return it->second.size();
    }
};




















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
    virtual void execute(SimulationContext &ctx) = 0;
    virtual std::string name() const = 0;
};

// -----------------------------------------------------------------------------
// 5. Konkretne Klasy Kroków Symulacji
// -----------------------------------------------------------------------------

// Krok 1: Iniekcja cząstek do układu
class ParticleInjectorStep : public ISimulationStep {
private:
    SpeciesID species_id_;
    float number_density_; // Fizyczna gęstość n [m^-3]
    float volume_;         // Objętość obszaru wstrzykiwania [m^3]

public:
    ParticleInjectorStep(SpeciesID species, float density, float volume)
        : species_id_(species), number_density_(density), volume_(volume) {}

    void execute(SimulationContext &ctx) override {
        const auto &traits = ctx.species_registry.get(species_id_);
        
        // Obliczenie liczby makrocząstek do stworzenia na podstawie wagi w
        double total_physical_particles = number_density_ * volume_;
        size_t macro_particles_to_inject = static_cast<size_t>(total_physical_particles / traits.weight);

        if (macro_particles_to_inject > 0) {
            auto handles = ctx.storage.allocate_space(species_id_, macro_particles_to_inject);
            
            std::cout << "  [" << name() << "] Dodano " << macro_particles_to_inject 
                      << " makrocząstek gatunku " << traits.name 
                      << " (waga w = " << traits.weight << ")\n";
        }
    }

    std::string name() const override { return "Particle Injector"; }
};

// Krok 2: Depozycja ładunku z cząstek na siatkę (Cloud-in-Cell)
// Krok 2: Depozycja ładunku z cząstek na siatkę (Cloud-in-Cell)
class ChargeDepositionStep : public ISimulationStep {
public:
    void execute(SimulationContext &ctx) override {
        ctx.grid.reset_charge_density(ctx.queue);

        for (SpeciesID id : ctx.species_registry.get_all_ids()) {
            const auto &traits = ctx.species_registry.get(id);
            float q_macro = traits.macro_charge(); // Ładunek całej makrocząstki

            // Pobieramy widoki SoA dla poszczególnych chunków
            auto views = ctx.storage.get_views(id);
            for (const auto &view : views) {
                // Jądro SYCL: Depozycja q_macro na węzły siatki dla view.count cząstek
                ctx.queue.parallel_for(sycl::range<1>(view.count), [=](sycl::id<1> idx) {
                    // Dostęp do pozycji: view.px[idx], view.py[idx], view.pz[idx]
                });
            }
            std::cout << "  [" << name() << "] Zdeponowano ładunek q_macro = " 
                      << q_macro << " C dla " << traits.name << "\n";
        }
    }

    std::string name() const override { return "Charge Deposition (CIC)"; }
};

// Krok 3: Solwer Pól (Poisson / Yee)
class FieldSolverStep : public ISimulationStep {
public:
    void execute(SimulationContext &ctx) override {
        // Solwer na siatce grid
        std::cout << "  [" << name() << "] Obliczono pola E oraz B na siatce.\n";
    }

    std::string name() const override { return "Field Solver"; }
};

// Krok 4: Pychacz Cząstek (Boris Pusher)
// Krok 4: Pychacz Cząstek (Boris Pusher)
class ParticlePusherStep : public ISimulationStep {
public:
    void execute(SimulationContext &ctx) override {
        for (SpeciesID id : ctx.species_registry.get_all_ids()) {
            const auto &traits = ctx.species_registry.get(id);
            float q_over_m = traits.q_over_m(); // Stosunek q/m cząstki

            // Pobieramy widoki SoA dla poszczególnych chunków
            auto views = ctx.storage.get_views(id);
            for (const auto &view : views) {
                float dt = ctx.dt;
                // Jądro SYCL: Algorytm Borisa z użyciem q_over_m i dt
                ctx.queue.parallel_for(sycl::range<1>(view.count), [=](sycl::id<1> idx) {
                    // Aktualizacja prędkości i pozycji wewnątrz view.vx[idx], view.px[idx] itp.
                });
            }
            std::cout << "  [" << name() << "] Przesunięto cząstki " << traits.name 
                      << " (q/m = " << q_over_m << " C/kg)\n";
        }
    }

    std::string name() const override { return "Particle Pusher (Boris)"; }
};

// Krok 5: Zderzenia cząstek (Monte Carlo Collisions - MCC)
class ParticleColliderStep : public ISimulationStep {
public:
    void execute(SimulationContext &ctx) override {
        std::cout << "  [" << name() << "] Przeliczono losowe zderzenia cząstek.\n";
    }

    std::string name() const override { return "Particle Collider (MCC)"; }
};

// -----------------------------------------------------------------------------
// 6. Główna Klasa Silnika Symulacji (SimulationEngine)
// -----------------------------------------------------------------------------

class SimulationEngine {
private:
    SimulationContext ctx_;
    std::vector<std::unique_ptr<ISimulationStep>> steps_;

public:
    SimulationEngine(sycl::queue &q, ParticleStorage &storage, GridField &grid, SpeciesRegistry &reg, float dt)
        : ctx_{q, storage, grid, reg, dt} {}

    // Rejestracja kroku przekazując wskaźnik unique_ptr
    void add_step(std::unique_ptr<ISimulationStep> step) {
        steps_.push_back(std::move(step));
    }

    // Rejestracja kroku szablonym wywołaniem (wygodne)
    template <typename T, typename... Args>
    void add_step(Args&&... args) {
        steps_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    // Wykonanie określonej liczby kroków czasowych
    void run(size_t total_steps) {
        std::cout << "=========================================================\n";
        std::cout << " Uruchomienie SimulationEngine (" << steps_.size() << " kroków w potoku)\n";
        std::cout << "=========================================================\n";

        for (size_t step = 0; step < total_steps; ++step) {
            std::cout << "\n---> Krok " << ctx_.current_step << " | t = " << ctx_.current_time << " s <---\n";

            for (auto &step_ptr : steps_) {
                step_ptr->execute(ctx_);
            }

            // Synchronizacja kolejki GPU na koniec kroku
            ctx_.queue.wait();

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
