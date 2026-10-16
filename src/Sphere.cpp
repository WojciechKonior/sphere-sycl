#include "ParticleStorage.hpp"
#include "GridField.hpp"
#include "Species.hpp"
#include "SimulationEngine.hpp"
#include "SimulationSteps/ParticleInjectorStep.hpp"
#include "SimulationSteps/ChargeDepositionStep.hpp"
#include "SimulationSteps/FieldSolverStep.hpp"
#include "SimulationSteps/ParticlePusherStep.hpp"
#include "SimulationSteps/ParticleColliderStep.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/PointSourceStrategy.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/BeamSourceStrategy.hpp"

int main() {
    sycl::queue q;

    // 1. Inicjalizacja składowych
    ParticleStorage storage(q);
    GridField grid(q);
    SpeciesRegistry species_reg;

    // 3. Utworzenie silnika symulacji (dt = 1 ps)
    SimulationEngine engine(q, storage, grid, species_reg, 1.0e-12f);

    // 1. Dodanie źródła punktowego ze sferycznym rozproszeniem o promieniu 0.05m i temperaturą 300K
    engine.add_step<ParticleInjectorStep>(
        SpeciesID::ELECTRON, 
        1.0e18 /* n */, 
        1.0e-6 /* V */,
        std::make_unique<PointSourceStrategy<MaxwellBoltzmannDistribution>>(
            Vector3{0.0f, 0.0f, 0.0f}, 
            1000.0 /* mean_v */, 
            0.05 /* source_sphere_radius */,
            MaxwellBoltzmannDistribution(300.0)
        )
    );

    // 2. Dodanie źródła strumieniowego (Beam / Ściana) o dużym promieniu tarczy (np. ściana)
    engine.add_step<ParticleInjectorStep>(
        SpeciesID::HYDROGEN_ION, 
        5.0e17 /* n */, 
        2.0e-6 /* V */, 
        std::make_unique<BeamSourceStrategy<MaxwellBoltzmannDistribution>>(
            Vector3{0.0f, 0.0f, -1.0f} /* center */, 
            Vector3{0.0f, 0.0f, 5000.0f} /* velocity dir */, 
            0.5 /* beam_radius / ściana */, 
            1000.0,
            MaxwellBoltzmannDistribution(1000.0)
        )
    );

    engine.add_step<ChargeDepositionStep>();
    engine.add_step<FieldSolverStep>();
    engine.add_step<ParticlePusherStep>();
    engine.add_step<ParticleColliderStep>();

    // 5. Uruchomienie symulacji
    engine.run(2);

    return 0;
}


