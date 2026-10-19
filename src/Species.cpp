#include "Species.hpp"
#include "SimulationUtils.hpp"

double SpeciesTraits::macro_mass() const { 
    return elem_mass * weight; 
}

double SpeciesTraits::macro_charge() const { 
    return elem_charge * weight; 
}

double SpeciesTraits::q_over_m() const { 
    return elem_charge / elem_mass; 
}

SpeciesRegistry::SpeciesRegistry() {
    // Automatyczna rejestracja standardowych gatunków plazmy przy tworzeniu obiektu
    register_species(
        SpeciesID::ELECTRON, 
        "Electron(e-)", 
        Constants::MASS_ELECTRON, 
        Constants::CHARGE_ELECTRON, 
        Constants::MACROPARTICLE_WEIGHT
    );

    register_species(
        SpeciesID::HYDROGEN_ION, 
        "Hydrogen ion(H+)", 
        Constants::MASS_HYDROGEN_ION, 
        Constants::CHARGE_HYDROGEN_ION, 
        Constants::MACROPARTICLE_WEIGHT
    );

    register_species(
        SpeciesID::NEUTRAL_HYDROGEN, 
        "Neutral hydrogen(H)", 
        Constants::MASS_HYDROGEN, 
        Constants::CHARGE_NEUTRAL, 
        Constants::MACROPARTICLE_WEIGHT
    );
}

void SpeciesRegistry::register_species(SpeciesID id, const std::string &name, double m, double q, double weight) {
    registry_[id] = SpeciesTraits{id, name, m, q, weight};
}

const SpeciesTraits& SpeciesRegistry::get(SpeciesID id) const {
    return registry_.at(id);
}

std::vector<SpeciesID> SpeciesRegistry::get_all_ids() const {
    std::vector<SpeciesID> ids;
    for (const auto &[id, traits] : registry_) {
        ids.push_back(id);
    }
    return ids;
}