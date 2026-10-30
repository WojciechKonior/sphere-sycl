#pragma once

#include <vector>
#include "IMesh.hpp"

/**
 * @brief Handles loading and storing unstructured mesh data for the PIC engine.
 */
class Msh : public IMesh {
    sycl::queue& queue_;
    MeshDeviceView mesh_device_view;

public:
    std::vector<Vector3> nodes;                  ///< List of mesh nodes (coordinates)
    Elements elements;                          ///< Struct of lists of 3D elements
    std::vector<MeshEntity> entities;            ///< List of mesh geometric entities

    /// @brief Construct a new Mesh object from a given file.
    explicit Msh(const std::string& filename, sycl::queue& q);
    ~Msh() override;

    /// @brief Returns the total number of nodes in the mesh.
    size_t get_node_count() const override ;

    /// @brief Returns the total number of 3D elements (tetrahedra) in the mesh.
    size_t get_element_count(ElementType type) const override ;

    /// @brief Returns a pointer to the raw, contiguous array of node coordinates.
    const Vector3* get_nodes_data() const override ;

    /// @brief Returns a pointer to the flattened, packed array of element node indices.
    const int* get_elements_data(ElementType type) const override ;

    /// @brief Returns the fixed number of nodes per element (4 for tetrahedra).
    int get_nodes_per_element(ElementType type) const override ;

    /// @brief Returns the device view containing raw GPU pointers and sizes.
    MeshDeviceView get_device_view() const override;
    
private:
    /// @brief Central dispatcher method. Responsible for identifying the file type and routing it to the appropriate parser.
    void load(const std::string& filename);
    
    /// @brief Inspects the header of the mesh file to verify if it's MSH v4 ASCII.
    bool is_msh_v4_ascii(const std::string& filename);
    
    /// @brief Parses Gmsh MSH v4.x ASCII mesh files.
    void load_msh_v4_ascii(const std::string& filename);
    
    /// @brief Parses the $Entities section containing geometric topology.
    void parse_entities(std::ifstream& file);
    
    /// @brief Parses the $Nodes section containing mesh node coordinates.
    void parse_nodes(std::ifstream& file);
    
    /// @brief Parses the $Elements section and populates homogeneous groups in elements_ structure.
    void parse_elements(std::ifstream& file);
    
    /// @brief Allocates USM device memory and copies mesh data from host to GPU.
    void upload_to_device();
};