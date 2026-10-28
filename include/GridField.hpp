#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <memory>

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

/**
 * @brief Handles loading and storing unstructured mesh data for the PIC engine.
 */
class Msh : public IMesh {
public:
    std::vector<Vector3> nodes;                  ///< List of mesh nodes (coordinates)
    std::vector<std::vector<int>> elements;      ///< List of 3D elements (tetrahedra defined by node indices)
    std::vector<MeshEntity> entities;            ///< List of mesh geometric entities

    /// @brief Construct a new Mesh object from a given file.
    explicit Msh(const std::string& filename) { load(filename); }
    ~Msh() override = default;

    /// @brief Returns the total number of nodes in the mesh.
    size_t get_node_count() const override {
        return nodes.size();
    }

    /// @brief Returns the total number of 3D elements (tetrahedra) in the mesh.
    size_t get_element_count() const override {
        // Since each tetrahedron consists of exactly 4 nodes
        return elements.size();
    }

    /// @brief Returns a pointer to the raw, contiguous array of node coordinates.
    const Vector3* get_nodes_data() const override {
        return nodes.data();
    }

    /// @brief Returns a pointer to the flattened, packed array of element node indices.
    const std::vector<int>* get_elements_data() const override {
        return elements.data();
    }

    /// @brief Returns the fixed number of nodes per element (4 for tetrahedra).
    int get_nodes_per_element(size_t element_index) const override {
        return elements.at(element_index).size(); // Tetrahedra
    }

private:
    /// @brief Central dispatcher method. Responsible for identifying the file type and routing it to the appropriate parser.
    void load(const std::string& filename) {
        if (is_msh_v4_ascii(filename)) {
            std::cout << "[Mesh] Detected format: MSH v4 (ASCII). Loading...\n";
            load_msh_v4_ascii(filename);
            return;
        } else {
            throw std::runtime_error("[Mesh] Error: Only MSH v4 ASCII format is currently supported for .msh files.");
        }
    }

    /// @brief Inspects the header of the mesh file to verify if it's MSH v4 ASCII.
    bool is_msh_v4_ascii(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("[Mesh] Failed to open file for validation: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("$MeshFormat") != std::string::npos) {
                double version = 0.0;
                int file_type = -1; // 0 = ASCII, 1 = Binary
                int data_size = 0;

                if (file >> version >> file_type >> data_size) {
                    if (version >= 4.0 && version < 5.0 && file_type == 0) {
                        return true;
                    }
                }
                break;
            }
        }
        return false;
    }

    /// @brief Parses Gmsh MSH v4.x ASCII mesh files.
    void load_msh_v4_ascii(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("[Mesh] Failed to open file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line == "$Entities") {
                parse_entities(file);
            }
            else if (line == "$Nodes") {
                parse_nodes(file);
            }
            else if (line == "$Elements") {
                parse_elements(file);
            }
        }
        
        file.close();
        std::cout << "[Mesh] Successfully loaded MSH v4.1 -> Nodes: " << nodes.size() 
                  << ", 3D Elements (Tetrahedra): " << elements.size() 
                  << ", Entities: " << entities.size() << "\n";
    }

    /// @brief Parses the $Entities section containing geometric topology.
    void parse_entities(std::ifstream& file) {
        size_t numPoints, numCurves, numSurfaces, numVolumes;
        file >> numPoints >> numCurves >> numSurfaces >> numVolumes;

        // Points (0D)
        for (size_t i = 0; i < numPoints; ++i) {
            MeshEntity ent;
            ent.dim = 0;
            double x, y, z;
            size_t numPhys;
            file >> ent.tag >> x >> y >> z >> numPhys;
            ent.bbox[0] = x; ent.bbox[1] = y; ent.bbox[2] = z;
            ent.bbox[3] = x; ent.bbox[4] = y; ent.bbox[5] = z;
            
            ent.physicalTags.resize(numPhys);
            for (size_t p = 0; p < numPhys; ++p) file >> ent.physicalTags[p];
            entities.push_back(ent);
        }

        // Curves (1D)
        for (size_t i = 0; i < numCurves; ++i) {
            MeshEntity ent;
            ent.dim = 1;
            size_t numPhys, numBoundingPoints;
            file >> ent.tag >> ent.bbox[0] >> ent.bbox[1] >> ent.bbox[2] 
                             >> ent.bbox[3] >> ent.bbox[4] >> ent.bbox[5] >> numPhys;
            
            ent.physicalTags.resize(numPhys);
            for (size_t p = 0; p < numPhys; ++p) file >> ent.physicalTags[p];
            
            file >> numBoundingPoints;
            for (size_t bp = 0; bp < numBoundingPoints; ++bp) {
                int dummy_pt; file >> dummy_pt;
            }
            entities.push_back(ent);
        }

        // Surfaces (2D)
        for (size_t i = 0; i < numSurfaces; ++i) {
            MeshEntity ent;
            ent.dim = 2;
            size_t numPhys, numBoundingCurves;
            file >> ent.tag >> ent.bbox[0] >> ent.bbox[1] >> ent.bbox[2] 
                             >> ent.bbox[3] >> ent.bbox[4] >> ent.bbox[5] >> numPhys;
            
            ent.physicalTags.resize(numPhys);
            for (size_t p = 0; p < numPhys; ++p) file >> ent.physicalTags[p];
            
            file >> numBoundingCurves;
            for (size_t bc = 0; bc < numBoundingCurves; ++bc) {
                int dummy_crv; file >> dummy_crv;
            }
            entities.push_back(ent);
        }

        // Volumes (3D)
        for (size_t i = 0; i < numVolumes; ++i) {
            MeshEntity ent;
            ent.dim = 3;
            size_t numPhys, numBoundingSurfaces;
            file >> ent.tag >> ent.bbox[0] >> ent.bbox[1] >> ent.bbox[2] 
                             >> ent.bbox[3] >> ent.bbox[4] >> ent.bbox[5] >> numPhys;
            
            ent.physicalTags.resize(numPhys);
            for (size_t p = 0; p < numPhys; ++p) file >> ent.physicalTags[p];
            
            file >> numBoundingSurfaces;
            for (size_t bs = 0; bs < numBoundingSurfaces; ++bs) {
                int dummy_srf; file >> dummy_srf;
            }
            entities.push_back(ent);
        }
    }

    /// @brief Parses the $Nodes section containing mesh node coordinates.
    void parse_nodes(std::ifstream& file) {
        size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;
        file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;
        
        nodes.resize(numNodes);

        for (size_t b = 0; b < numEntityBlocks; ++b) {
            int entityDim, entityTag, parametric, numNodesInBlock;
            file >> entityDim >> entityTag >> parametric >> numNodesInBlock;

            std::vector<size_t> blockNodeTags(numNodesInBlock);
            for (int i = 0; i < numNodesInBlock; ++i) {
                file >> blockNodeTags[i];
            }

            for (int i = 0; i < numNodesInBlock; ++i) {
                double x, y, z;
                file >> x >> y >> z;
                
                // Convert 1-based Gmsh tag to 0-based vector index
                size_t global_idx = blockNodeTags[i] - 1;
                if (global_idx < nodes.size()) {
                    nodes[global_idx] = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
                }
            }
        }
    }

    /// @brief Parses the $Elements section, filtering out 3D volumetric tetrahedra.
    void parse_elements(std::ifstream& file) {
        size_t numEntityBlocks, numElements, minElementTag, maxElementTag;
        file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;

        for (size_t b = 0; b < numEntityBlocks; ++b) {
            int entityDim, entityTag, elementType;
            size_t numElementsInBlock;
            file >> entityDim >> entityTag >> elementType >> numElementsInBlock;

            // Element type 4 corresponds to a 4-node tetrahedron in Gmsh
            bool is_tetrahedron = (elementType == 4);

            for (size_t i = 0; i < numElementsInBlock; ++i) {
                size_t elemTag;
                file >> elemTag;

                int nodes_per_elem = (elementType == 4) ? 4 : 0;

                std::vector<int> elem_nodes(nodes_per_elem);
                for (int n = 0; n < nodes_per_elem; ++n) {
                    size_t nodeTag;
                    file >> nodeTag;
                    elem_nodes[n] = static_cast<int>(nodeTag) - 1; // Convert to 0-based indexing
                }

                if (is_tetrahedron) {
                    elements.push_back(elem_nodes);
                }
            }
        }
    }
};

struct GridFieldDeviceView {
    const float* nodes_x;
    const float* nodes_y;
    const float* nodes_z;
    int num_nodes;

    inline bool device_out_of_boundary_check(float x, float y, float z) const { return false; }
};


/// @brief Represents spatial grid fields (e.g., charge density, electric and magnetic fields) for the simulation.
class GridField
{
private:
    sycl::queue &queue_;
    std::unique_ptr<IMesh> mesh_;
    // sycl::buffer<float, 1> rho_buffer_;
    GridFieldDeviceView device_view_;

public:
    explicit GridField(sycl::queue &q, const std::string& filename = "EMPTY") 
        : queue_(q), device_view_() 
        {
            if(filename != "EMPTY")
            {
                std::filesystem::path filePath(filename);
                std::string ext = filePath.extension().string();
    
                // Obsługa rozszerzenia .msh
                if (ext == ".msh") {
                    mesh_ = std::make_unique<Msh>(filename);
                }
                
                // --- TUTAJ W PRZYSŁOŚCI MOŻESZ DODAĆ KOLEJNE FORMATY ---
                // else if (ext == ".vtk") { load_vtk(filename); return; }
                // else if (ext == ".fem") { load_fem(filename); return; }
    
                // Jeśli rozszerzenie nie pasuje do żadnego znanego handlera:
                throw std::runtime_error("[Mesh] Error: Unknown or unsupported file extension: " + ext);
            }
        }

    /// @brief Zwraca kolejkę SYCL przypisaną do siatki
    sycl::queue& get_queue() { return queue_; }

    /// @brief Zwraca lekki widok urządzenia do przekazania do kernela SYCL
    GridFieldDeviceView get_device_view() const { return device_view_; }

    const std::unique_ptr<IMesh>& get_mesh() const { return mesh_; }
};