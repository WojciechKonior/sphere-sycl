#pragma once

#include <sycl/sycl.hpp>

/// @brief Represents spatial grid fields (e.g., charge density, electric and magnetic fields) for the simulation.
class GridField
{
public:
    explicit GridField(sycl::queue &q);

    /// @brief Resets the charge density array on the device (sets values to zero) prior to deposition.
    void reset_charge_density(sycl::queue &q);
};