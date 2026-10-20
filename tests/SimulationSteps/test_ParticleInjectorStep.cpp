#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "Species.hpp"
#include "SimulationSteps/ParticleInjectorStep.hpp"
#include "SimulationSteps/ParticleInjectorStrategy/IParticleInjectorStrategy.hpp"
#include "GridField.hpp"

// Pomocnicza strategia mockująca do testów analitycznych
class MockInjectorStrategy : public IParticleInjectorStrategy {
private:
    double surface_area_;
    double characteristic_velocity_;

public:
    MockInjectorStrategy(double surface_area, double characteristic_velocity)
        : surface_area_(surface_area), characteristic_velocity_(characteristic_velocity) {}

    double get_surface_area() const override {
        return surface_area_;
    }

    double get_characteristic_velocity() const override {
        return characteristic_velocity_;
    }

    size_t compute_macro_particles_to_inject(const SimulationContext &ctx, SpeciesID species_id, double number_density) const override {
        if (ctx.dt <= 0.0 || number_density <= 0.0) 
            return 0;
        const auto &traits = ctx.species_registry.get(species_id);
        double dt = static_cast<double>(ctx.dt);
        double total_physical_particles = number_density * characteristic_velocity_ * surface_area_ * dt;
        double macro_particles_to_inject = total_physical_particles / static_cast<double>(traits.weight);
        return static_cast<size_t>(std::lround(macro_particles_to_inject));
    }

    void device_initialize_particles(const SegmentHandle &handle, const InjectionParameters &params, sycl::handler &cgh) const override {}
};

/**
 * @class ParticleInjectorStepTest
 * @brief Test fixture for ParticleInjectorStep unit tests.
 */
class ParticleInjectorStepTest : public ::testing::Test {
protected:
    sycl::queue queue_;
    ParticleStorage storage_;
    GridField grid_;
    SpeciesRegistry species_registry_;

    ParticleInjectorStepTest() 
        : storage_(queue_), grid_(queue_) {}

    void SetUp() override {
        // Inicjalizacja środowiska testowego
    }

    void TearDown() override {
        // Czyszczenie po testach
    }
};

/**
 * @test Verifies correct calculation of macroparticles to inject under standard physical parameters
 *       using the default pre-registered electron species.
 */
TEST_F(ParticleInjectorStepTest, ComputeMacroParticlesStandardFlow) {
    // 1. Pobieramy domyślną wagę elektronu z rejestru (zainicjalizowaną w konstruktorze SpeciesRegistry)
    const auto &electron_traits = species_registry_.get(SpeciesID::ELECTRON);
    double default_weight = electron_traits.weight; // np. 0.01

    // 2. Dobieramy parametry fizyczne tak, aby otrzymać np. dokładnie 100 makrocząstek
    // Wzór: macro_particles = (n * v * A * dt) / weight
    // Załóżmy cel: 100 makrocząstek -> N_fizyczne = 100 * default_weight
    double target_macro_particles = 100.0;
    double total_physical_particles = target_macro_particles * default_weight;

    // Rozbijamy N_fizyczne na czytelne parametry:
    double number_density = 1.0e15;                  // 1e15 m^-3
    double characteristic_velocity = 1000.0;         // 1000 m/s
    double surface_area = 1.0;                       // 1 m^2
    
    // Wyznaczamy taki dt, aby dał dokładnie wymaganą liczbę cząstek fizycznych:
    // dt = total_physical_particles / (number_density * velocity * area)
    double dt = total_physical_particles / (number_density * characteristic_velocity * surface_area);

    auto strategy = std::make_unique<MockInjectorStrategy>(surface_area, characteristic_velocity);
    
    // Używamy istniejącego w rejestrze SpeciesID::ELECTRON
    ParticleInjectorStep injector(SpeciesID::ELECTRON, number_density, std::move(strategy));

    SimulationContext ctx{queue_, storage_, grid_, species_registry_, dt};

    size_t result = injector.compute_macro_particles_to_inject(ctx);

    // Oczekujemy dokładnie tyle makrocząstek, ile zaplanowaliśmy
    EXPECT_EQ(result, static_cast<size_t>(target_macro_particles));
}

/**
 * @test Verifies that zero macroparticles are injected when time step dt or density is zero.
 */
TEST_F(ParticleInjectorStepTest, ComputeMacroParticlesZeroWhenDtOrDensityZero) {
    double number_density = 1.0e18;
    auto strategy = std::make_unique<MockInjectorStrategy>(1.0, 1000.0);
    
    ParticleInjectorStep injector(SpeciesID::ELECTRON, number_density, std::move(strategy));

    // Przypadek 1: dt = 0.0
    SimulationContext ctx_zero_dt{queue_, storage_, grid_, species_registry_, 0.0};
    EXPECT_EQ(injector.compute_macro_particles_to_inject(ctx_zero_dt), 0u);

    // Przypadek 2: gęstość = 0.0
    ParticleInjectorStep injector_zero_density(SpeciesID::ELECTRON, 0.0, std::make_unique<MockInjectorStrategy>(1.0, 1000.0));
    SimulationContext ctx_normal{queue_, storage_, grid_, species_registry_, 1.0e-12};
    EXPECT_EQ(injector_zero_density.compute_macro_particles_to_inject(ctx_normal), 0u);
}

/**
 * @test Verifies scaling behavior with macroparticle weight (heavier weight results in fewer macroparticles).
 */
TEST_F(ParticleInjectorStepTest, ComputeMacroParticlesWeightScaling) {
    double number_density = 1.0e18;
    double dt = 1.0e-6;
    auto strategy = std::make_unique<MockInjectorStrategy>(1.0, 1000.0);

    SpeciesID sp = SpeciesID::HYDROGEN_ION;

    // Test z mniejszą wagą
    species_registry_.register_species(sp, "ProtonLight", Constants::MASS_HYDROGEN_ION, Constants::CHARGE_HYDROGEN_ION, 1.0e6);
    ParticleInjectorStep injector_light(sp, number_density, std::make_unique<MockInjectorStrategy>(1.0, 1000.0));
    SimulationContext ctx{queue_, storage_, grid_, species_registry_, dt};
    size_t count_light = injector_light.compute_macro_particles_to_inject(ctx);

    // Test z 10x większą wagą
    species_registry_.register_species(sp, "ProtonHeavy", Constants::MASS_HYDROGEN_ION, Constants::CHARGE_HYDROGEN_ION, 1.0e7);
    ParticleInjectorStep injector_heavy(sp, number_density, std::make_unique<MockInjectorStrategy>(1.0, 1000.0));
    size_t count_heavy = injector_heavy.compute_macro_particles_to_inject(ctx);

    // 10x większa waga powinna dać dokładnie 10x mniej makrocząstek
    EXPECT_EQ(count_light, count_heavy * 10u);
}

