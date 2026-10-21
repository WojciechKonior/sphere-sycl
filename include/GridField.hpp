#pragma once

#include <sycl/sycl.hpp>

struct GridFieldDeviceView {
    inline bool device_out_of_boundary_check(float x, float y, float z) const { return false; }
};

/// @brief Represents spatial grid fields (e.g., charge density, electric and magnetic fields) for the simulation.
class GridField
{
private:
    sycl::queue &queue_;
    GridFieldDeviceView device_view_;

public:
    explicit GridField(sycl::queue &q) : queue_(q), device_view_() {}

    /// @brief Zwraca kolejkę SYCL przypisaną do siatki
    sycl::queue& get_queue() { return queue_; }

    /// @brief Zwraca lekki widok urządzenia do przekazania do kernela SYCL
    GridFieldDeviceView get_device_view() const { return device_view_; }
};