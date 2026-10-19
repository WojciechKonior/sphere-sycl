#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>

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
    
    double elem_mass;    ///< Elementary particle mass [kg]
    double elem_charge;  ///< Elementary particle charge [C]
    double weight;       ///< Numerical macroparticle weight (number of physical particles per macroparticle)

    /// @return Mass of a single macroparticle [kg].
    double macro_mass() const;

    /// @return Charge of a single macroparticle [C].
    double macro_charge() const;

    /// @return Charge-to-mass ratio q/m (identical for elementary and macroparticles).
    double q_over_m() const;
};

/// @brief Registry for managing and retrieving traits of active species in the simulation.
class SpeciesRegistry {
private:
    std::unordered_map<SpeciesID, SpeciesTraits> registry_;

public:
    /// @brief Constructor that automatically populates the registry with standard physical species.
    explicit SpeciesRegistry();

    /// @brief Registers a new species or updates an existing entry in the registry.
    void register_species(SpeciesID id, const std::string &name, double m, double q, double weight);

    /// @brief Retrieves traits for a given species ID.
    const SpeciesTraits& get(SpeciesID id) const;

    /// @return List of all registered species IDs.
    std::vector<SpeciesID> get_all_ids() const;
};