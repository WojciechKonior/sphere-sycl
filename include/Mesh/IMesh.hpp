#pragma once

#include <vector>
#include <array>
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

/**
 * @brief A lightweight device view struct for handling a specific group of mesh elements on the GPU.
 */
struct ElementsDeviceView {
    const int* indices = nullptr;           ///< Pointer to element node indices allocated on the device memory.
    size_t num_indices = 0;                 ///< Total number of indices stored in the indices array.

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
    ElementsDeviceView& elements_at(ElementType type) {
        return d_elements[type.to_index()];
    }
    
    /// @brief Accesses the ElementsDeviceView for a given ElementType (const version).
    const ElementsDeviceView& elements_at(ElementType type) const {
        return d_elements[type.to_index()];
    }

    /// @brief Accesses the ElementsDeviceView using a raw Gmsh integer ID.
    ElementsDeviceView& elements_at(int raw_gmsh_id) {
        return d_elements[ElementType(raw_gmsh_id).to_index()];
    }
    
    /// @brief Accesses the ElementsDeviceView using a raw Gmsh integer ID (const version).
    const ElementsDeviceView& elements_at(int raw_gmsh_id) const {
        return d_elements[ElementType(raw_gmsh_id).to_index()];
    }

    /// @brief Return true if (x,y,z) point is out of boundary.
    inline bool device_out_of_boundary_check(auto x, auto y, auto z) const { return false; }

    /// @brief Allocates USM device memory and computes volume for each element on GPU.
    void compute_element_volumes(sycl::queue& q)
    {
        for (ElementType type : ElementType::ALL) {
            auto &d_elements = elements_at(type);
            
            // Jeśli dany typ elementów nie występuje w siatce, pomijamy
            if (d_elements.num_indices == 0 || d_elements.nodes_per_element == 0) {
                d_elements.volumes = nullptr;
                continue;
            }

            size_t num_elements = d_elements.num_indices / d_elements.nodes_per_element;

            // 1. Alokacja pamięci USM na urządzeniu dla objętości elementów
            float* d_volumes_mutable = sycl::malloc_device<float>(num_elements, q);
            d_elements.volumes = d_volumes_mutable; // Przypisanie doconst wskaźnika w widoku

            // Pobieramy stałe widoki do przekazania przez wartość do lambdy SYCL
            const Vector3* d_nodes_ptr = d_nodes.nodes;
            const int* d_indices_ptr = d_elements.indices;
            size_t nodes_per_elem = d_elements.nodes_per_element;
            ElementType current_type = type;

            // 2. Uruchomienie kernela SYCL za pomocą parallel_for
            q.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1> idx) {
                size_t e = idx[0];
                size_t offset = e * nodes_per_elem;
                d_volumes_mutable[e] = current_type.device_compute_element_volume(d_nodes_ptr, d_indices_ptr);
            }).wait(); // Czekamy na zakończenie obliczeń kernela
        }
    }

    /// @brief Allocates USM device memory and computes centroids for each element on GPU.
    void compute_element_centroids(sycl::queue& q){
        for (ElementType type : ElementType::ALL) {
            auto &d_elements = elements_at(type);
            
            if (d_elements.num_indices == 0 || d_elements.nodes_per_element == 0) {
                d_elements.centroids = nullptr;
                continue;
            }

            size_t num_elements = d_elements.num_indices / d_elements.nodes_per_element;

            // 1. Alokacja pamięci USM na urządzeniu dla środków ciężkości
            Vector3* d_centroids_mutable = sycl::malloc_device<Vector3>(num_elements, q);
            d_elements.centroids = d_centroids_mutable;

            const Vector3* d_nodes_ptr = d_nodes.nodes;
            const int* d_indices_ptr = d_elements.indices;
            size_t nodes_per_elem = d_elements.nodes_per_element;
            ElementType current_type = type;

            // 2. Uruchomienie kernela SYCL
            q.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1> idx) {
                size_t e = idx[0];
                const int* element_indices = d_indices_ptr + (e * nodes_per_elem);
                d_centroids_mutable[e] = current_type.device_compute_element_centroid(d_nodes_ptr, element_indices);
            }).wait();
        }
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
