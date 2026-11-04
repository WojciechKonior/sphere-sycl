#pragma once

#include <array>
#include "SimulationUtils.hpp"

/**
 * @brief A lightweight device view struct for handling a specific group of mesh elements on the GPU.
 */
struct ElementsDeviceView {
    const int* indices = nullptr;           ///< Pointer to element node indices allocated on the device memory.
    size_t num_indices = 0;                 ///< Total number of indices stored in the indices array.
    size_t num_elements = 0;                ///< Total number of elements of this type stored in this container.

    const float* volumes = nullptr;         ///< The volume for each element in this group.
    const Vector3* centroids = nullptr;     ///< Centers of mass of elements.

    ElementType type;                       ///< The geometrical element type of this group.
    size_t nodes_per_element = 0;           ///< Number of nodes defining a single element (e.g., 4 for tetrahedra).
};

/**
 * @brief A lightweight device view struct for handling a nodes data on the GPU.
 */
struct NodesDeviceView {
    const Vector3* nodes = nullptr;
    size_t num_nodes = 0;
};

/**
 * @brief A lightweight struct for handling mesh data on gpu.
 */
struct MeshDeviceView {
    NodesDeviceView d_nodes;
    std::array<ElementsDeviceView, ElementType::SIZE> d_elements;

    /// @brief Accesses the ElementsDeviceView for a given ElementType.
    ElementsDeviceView& elements_at(ElementType type);
    
    /// @brief Accesses the ElementsDeviceView for a given ElementType (const version).
    const ElementsDeviceView& elements_at(ElementType type) const;

    /// @brief Accesses the ElementsDeviceView using a raw Gmsh integer ID.
    ElementsDeviceView& elements_at(int raw_gmsh_id);
    
    /// @brief Accesses the ElementsDeviceView using a raw Gmsh integer ID (const version).
    const ElementsDeviceView& elements_at(int raw_gmsh_id) const;

    /// @brief Return true if (x,y,z) point is out of boundary.
    inline bool device_out_of_boundary_check(auto x, auto y, auto z) const { return false; }

    /// @brief Allocates USM device memory and computes volume for each element on GPU.
    void compute_element_volumes(sycl::queue& q);

    /// @brief Allocates USM device memory and computes centroids for each element on GPU.
    void compute_element_centroids(sycl::queue& q);
};