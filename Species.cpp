#include "Species.hpp"

float SpeciesTraits::macro_mass() const { 
    return elem_mass * weight; 
}

float SpeciesTraits::macro_charge() const { 
    return elem_charge * weight; 
}

float SpeciesTraits::q_over_m() const { 
    return elem_charge / elem_mass; 
}

void SpeciesRegistry::register_species(SpeciesID id, const std::string &name, float m, float q, float weight) {
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