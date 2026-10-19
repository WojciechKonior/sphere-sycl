#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "Species.hpp"
#include "SimulationUtils.hpp"

/**
 * @class SpeciesTraitsTest
 * @brief Test fixture for SpeciesTraits and SpeciesRegistry class unit tests.
 */
class SpeciesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialization if needed
    }

    void TearDown() override {
        // Cleanup if needed
    }
};

/**
 * @test Verifies correct calculation of macroparticle mass, charge, and q/m ratio for an Electron.
 */
TEST_F(SpeciesTest, ElectronMacroPropertiesCalculation) {
    SpeciesTraits electron{
        SpeciesID::ELECTRON,
        "Elektron",
        Constants::MASS_ELECTRON,
        Constants::CHARGE_ELECTRON,
        Constants::MACROPARTICLE_WEIGHT
    };

    // Macro mass should be elementary mass multiplied by weight
    double expected_macro_mass = Constants::MASS_ELECTRON * Constants::MACROPARTICLE_WEIGHT;
    EXPECT_NEAR(electron.macro_mass(), expected_macro_mass, 1e-35);

    // Macro charge should be elementary charge multiplied by weight
    double expected_macro_charge = Constants::CHARGE_ELECTRON * Constants::MACROPARTICLE_WEIGHT;
    EXPECT_NEAR(electron.macro_charge(), expected_macro_charge, 1e-15);

    // q/m ratio should remain identical for elementary and macroparticles
    double expected_q_over_m = Constants::CHARGE_ELECTRON / Constants::MASS_ELECTRON;
    EXPECT_NEAR(electron.q_over_m(), expected_q_over_m, 1e-5);
}

/**
 * @test Verifies correct calculation of properties for a Hydrogen Ion (Proton) and Neutral Hydrogen.
 */
TEST_F(SpeciesTest, IonAndNeutralPropertiesCalculation) {
    SpeciesTraits proton{
        SpeciesID::HYDROGEN_ION,
        "Proton",
        Constants::MASS_HYDROGEN_ION,
        Constants::CHARGE_HYDROGEN_ION,
        Constants::MACROPARTICLE_WEIGHT
    };

    EXPECT_GT(proton.macro_mass(), 0.0);
    EXPECT_GT(proton.macro_charge(), 0.0); // Positive charge for ion

    SpeciesTraits neutral{
        SpeciesID::NEUTRAL_HYDROGEN,
        "Neutralny Wodór",
        Constants::MASS_HYDROGEN,
        Constants::CHARGE_NEUTRAL,
        Constants::MACROPARTICLE_WEIGHT
    };

    EXPECT_GT(neutral.macro_mass(), 0.0);
    EXPECT_EQ(neutral.macro_charge(), 0.0); // Neutral charge must be zero
    EXPECT_EQ(neutral.q_over_m(), 0.0);     // q/m for neutral should be zero
}

/**
 * @test Verifies automatic initialization of standard species in SpeciesRegistry constructor.
 */
TEST_F(SpeciesTest, RegistryAutomaticInitialization) {
    SpeciesRegistry registry;

    // Verify all expected IDs are present
    std::vector<SpeciesID> ids = registry.get_all_ids();
    EXPECT_EQ(ids.size(), 3u);

    // Verify retrieval of individual species traits
    const auto &electron = registry.get(SpeciesID::ELECTRON);
    EXPECT_EQ(electron.name, "Electron(e-)");
    EXPECT_EQ(electron.elem_charge, Constants::CHARGE_ELECTRON);

    const auto &proton = registry.get(SpeciesID::HYDROGEN_ION);
    EXPECT_EQ(proton.name, "Hydrogen ion(H+)");
    EXPECT_EQ(proton.elem_mass, Constants::MASS_HYDROGEN_ION);

    const auto &neutral = registry.get(SpeciesID::NEUTRAL_HYDROGEN);
    EXPECT_EQ(neutral.name, "Neutral hydrogen(H)");
    EXPECT_EQ(neutral.elem_charge, Constants::CHARGE_NEUTRAL);
}

/**
 * @test Verifies ability to register custom species or override entries in the registry.
 */
TEST_F(SpeciesTest, RegistryCustomRegistration) {
    SpeciesRegistry registry;

    SpeciesID custom_id = static_cast<SpeciesID>(99);
    registry.register_species(custom_id, "CustomIon", 5.0e-26, 3.204e-19, 1.0e6);

    const auto &custom = registry.get(custom_id);
    EXPECT_EQ(custom.name, "CustomIon");
    EXPECT_FLOAT_EQ(custom.elem_mass, 5.0e-26);
    EXPECT_FLOAT_EQ(custom.weight, 1.0e6);
}

