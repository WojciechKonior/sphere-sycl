#pragma once

#include <vector>
#include "SimulationUtils.hpp"

/**
 * @brief Supported mesh file formats.
 */
enum class MeshFormat {
    MSH_v4_ASCII,
    Unknown
};

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
    // --- Elementy objętościowe 3D (dla domeny symulacji) ---
    std::vector<int> tetrahedra;    // Spakowane czworościany (stride = 4)
    std::vector<int> hexahedra;     // Spakowane sześciany (stride = 8)
    std::vector<int> prisms;        // Spakowane graniastosłupy trójkątne (stride = 6)
    std::vector<int> pyramids;      // Spakowane ostrosłupy (stride = 5)

    // --- Elementy powierzchniowe 2D (przydatne do warunków brzegowych) ---
    std::vector<int> boundary_triangles; // Trójkąty brzegowe (stride = 3)
    std::vector<int> boundary_quads;     // Czworokąty brzegowe (stride = 4)
};

struct IMesh{
    virtual ~IMesh() = default;
    
    /// @brief Returns the total number of nodes in the mesh.
    virtual size_t get_node_count() const = 0;

    /// @brief Returns the total number of volumetric elements in the mesh.
    virtual size_t get_element_count() const = 0;

    /// @brief Returns a raw pointer to the contiguous array of node coordinates.
    virtual const Vector3* get_nodes_data() const = 0;
    
    /// @brief Returns a raw pointer to the flattened, packed array of element node indices 
    virtual const std::vector<int>* get_elements_data() const = 0;

    /// @brief Returns the fixed number of nodes per element (e.g., 4 for tetrahedra).
    virtual int get_nodes_per_element(size_t element_index) const = 0;

    // --- 3. Boundary Conditions & Topology ---
    // virtual const std::vector<BoundaryPatch>& get_boundaries() const = 0;
};
