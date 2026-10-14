#include "ParticleStorage.hpp"
#include "GridField.hpp"
#include "Species.hpp"
#include "SimulationEngine.hpp"
#include "SimulationSteps/ParticleInjectorStep.hpp"
#include "SimulationSteps/ChargeDepositionStep.hpp"
#include "SimulationSteps/FieldSolverStep.hpp"
#include "SimulationSteps/ParticlePusherStep.hpp"
#include "SimulationSteps/ParticleColliderStep.hpp"

int main() {
    sycl::queue q;

    // 1. Inicjalizacja składowych
    ParticleStorage storage(q);
    GridField grid(q);
    SpeciesRegistry species_reg;

    // // 2. Rejestracja gatunków z wykorzystaniem przestrzeni Constants
    // species_reg.register_species(
    //     SpeciesID::ELECTRON, 
    //     "Elektron", 
    //     Constants::MASS_ELECTRON, 
    //     Constants::CHARGE_ELECTRON, 
    //     Constants::MACROPARTICLE_WEIGHT
    // );

    // species_reg.register_species(
    //     SpeciesID::HYDROGEN_ION, 
    //     "Proton", 
    //     Constants::MASS_HYDROGEN_ION, 
    //     Constants::CHARGE_HYDROGEN_ION, 
    //     Constants::MACROPARTICLE_WEIGHT
    // );

    // 3. Utworzenie silnika symulacji (dt = 1 ps)
    SimulationEngine engine(q, storage, grid, species_reg, 1.0e-12f);

    // 4. Budowanie potoku wykonywania (Pipeline)
    engine.add_step<ParticleInjectorStep>(SpeciesID::ELECTRON, 1.0e18f /* n */, 1.0e-6f /* V */);
    // // 1. Dodanie źródła punktowego ze sferycznym rozproszeniem o promieniu 0.05m i temperaturą 300K
    // engine.add_step<ParticleInjectorStep>(
    //     SpeciesID::ELECTRON, 
    //     1.0e18f /* n */, 
    //     1.0e-6f /* V */, 
    //     300.0f /* T [K] */,
    //     std::make_unique<PointSourceStrategy>(Vector3{0.0f, 0.0f, 0.0f}, 1000.0f /* mean_v */, 0.05f /* sphere_radius */)
    // );

    // // 2. Dodanie źródła strumieniowego (Beam / Ściana) o dużym promieniu tarczy (np. ściana)
    // engine.add_step<ParticleInjectorStep>(
    //     SpeciesID::HYDROGEN_ION, 
    //     5.0e17f /* n */, 
    //     2.0e-6f /* V */, 
    //     1000.0f /* T [K] */,
    //     std::make_unique<BeamSourceStrategy>(Vector3{0.0f, 0.0f, -1.0f} /* center */, Vector3{0.0f, 0.0f, 5000.0f} /* velocity dir */, 0.5f /* circle_radius / ściana */, 1000.0f)
    // );

    engine.add_step<ChargeDepositionStep>();
    engine.add_step<FieldSolverStep>();
    engine.add_step<ParticlePusherStep>();
    engine.add_step<ParticleColliderStep>();

    // 5. Uruchomienie symulacji
    engine.run(2);

    return 0;
}


