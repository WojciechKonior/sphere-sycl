#pragma once

#include <vector>
#include <array>
#include "SimulationUtils.hpp"
#include "MeshDeviceView.hpp"

/**
 * @brief Represents a geometric entity (point, curve, surface, or volume) from $Entities.
 */
struct MeshEntity {
    int dim;                         // Wymiar: 0 (punkt), 1 (krzywa), 2 (powierzchnia), 3 (objętość)
    int tag;                         // Unikalny numer/tag encji
    double bbox[6];                  // Bounding box: {xmin, ymin, zmin, xmax, ymax, zmax}
    std::vector<int> physicalTags;   // Tagi grup fizycznych (przydatne do warunków brzegowych)
};

/**
 * @brief Container holding homogeneous groups of mesh elements, 
 * optimized for flat memory layout and GPU execution.
 */
struct Elements {
private:
    std::array<std::vector<int>, static_cast<size_t>(ElementType::SIZE)> data;

public:
    /// @brief Accesses the element vector for a given ElementType.
    std::vector<int>& at(ElementType type) {
        return data[type.to_index()];
    }
    const std::vector<int>& at(ElementType type) const {
        return data[type.to_index()];
    }

    /// @brief Accesses the element vector using a raw Gmsh integer ID.
    std::vector<int>& at(int raw_gmsh_id) {
        return data[ElementType(raw_gmsh_id).to_index()];
    }
    const std::vector<int>& at(int raw_gmsh_id) const {
        return data[ElementType(raw_gmsh_id).to_index()];
    }
};

struct IMesh{
    virtual ~IMesh() = default;
    
    /// @brief Returns the total number of nodes in the mesh.
    virtual size_t get_node_count() const = 0;

    /// @brief Returns the total number of volumetric elements in the mesh.
    virtual size_t get_element_count(ElementType type) const = 0;

    /// @brief Returns a raw pointer to the contiguous array of node coordinates.
    virtual const Vector3* get_nodes_data() const = 0;
    
    /// @brief Returns a raw pointer to the flattened, packed array of element node indices 
    virtual const ElementsDeviceView& get_elements_data(ElementType type) const = 0;

    /// @brief Returns the fixed number of nodes per element (e.g., 4 for tetrahedra).
    virtual int get_nodes_per_element(ElementType type) const = 0;

    /// @brief Returns the device view containing raw GPU pointers and sizes.
    virtual MeshDeviceView get_device_view() const = 0;

    // --- 3. Boundary Conditions & Topology ---
    // virtual const std::vector<BoundaryPatch>& get_boundaries() const = 0;
};
