#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>

/// @brief Physical and simulation configuration constants.
namespace Constants {
    constexpr float MACROPARTICLE_WEIGHT = 1.0e8f;

    constexpr float MASS_ELECTRON = 9.109e-31f;
    constexpr float MASS_HYDROGEN_ION   = 1.672e-27f;
    constexpr float MASS_HYDROGEN = 1.674e-27f;

    constexpr float CHARGE_ELEMENTARY = 1.602e-19f;
    constexpr float CHARGE_ELECTRON   = -CHARGE_ELEMENTARY;
    constexpr float CHARGE_HYDROGEN_ION     =  CHARGE_ELEMENTARY;
    constexpr float CHARGE_NEUTRAL    =  0.0f;

    constexpr size_t CHUNK_CAPACITY = 10000000; // 1e7
}

/// @brief Identifiers for physical particle species used in the simulation.
enum class SpeciesID {
    ELECTRON,
    HYDROGEN_ION,
    NEUTRAL_HYDROGEN
};

/// @brief Defines the physical properties and numerical weighting of a particle species.
struct SpeciesTraits {
    SpeciesID id;
    std::string name;
    
    float elem_mass;    ///< Elementary particle mass [kg]
    float elem_charge;  ///< Elementary particle charge [C]
    float weight;       ///< Numerical macroparticle weight (number of physical particles per macroparticle)

    /// @return Mass of a single macroparticle [kg].
    float macro_mass() const;

    /// @return Charge of a single macroparticle [C].
    float macro_charge() const;

    /// @return Charge-to-mass ratio q/m (identical for elementary and macroparticles).
    float q_over_m() const;
};

/// @brief Registry for managing and retrieving traits of active species in the simulation.
class SpeciesRegistry {
private:
    std::unordered_map<SpeciesID, SpeciesTraits> registry_;

public:
    /// @brief Registers a new species or updates an existing entry in the registry.
    void register_species(SpeciesID id, const std::string &name, float m, float q, float weight);

    /// @brief Retrieves traits for a given species ID.
    const SpeciesTraits& get(SpeciesID id) const;

    /// @return List of all registered species IDs.
    std::vector<SpeciesID> get_all_ids() const;
};