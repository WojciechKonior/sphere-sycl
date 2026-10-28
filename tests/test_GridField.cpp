#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <filesystem>
#include "GridField.hpp"

/**
 * @class GridFieldTest
 * @brief Test fixture for GridFieldTest class unit tests.
 */
class GridFieldTest : public ::testing::Test {
protected:
    /// SYCL queue used for memory allocation and execution within tests.
    sycl::queue queue;

    /**
     * @brief Sets up the test environment by creating a default SYCL queue.
     */
    void SetUp() override {
        try {
            queue = sycl::queue(sycl::gpu_selector_v);
        } catch (const sycl::exception &e) {
            FAIL() << "Failed to create SYCL queue: " << e.what();
        }
    }

    /**
     * @brief Cleans up the test environment after each test execution.
     */
    void TearDown() override {
        try {
            queue.wait();
        } catch (const sycl::exception &e) {
            ADD_FAILURE() << "SYCL exception during TearDown wait: " << e.what();
        }
    }
};

/**
 * @test Verifies that the Mesh class correctly loads an unstructured .msh file 
 * from the assets directory and populates nodes and elements.
 */
TEST_F(GridFieldTest, DynamicFileParityVerification) {
    std::string filepath = std::string(TEST_ASSETS_DIR) + "/simple_box.msh";
    
    ASSERT_TRUE(std::filesystem::exists(filepath)) 
        << "Brak pliku siatki pod ścieżką: " << std::filesystem::absolute(filepath);

    // 1. Wczytujemy siatkę za pomocą naszej klasy
    Msh mesh(filepath);

    // 2. Otwieramy ten sam plik bezpośrednio w teście do weryfikacji na żywo
    std::ifstream file(filepath);
    ASSERT_TRUE(file.is_open());

    std::string line;
    while (std::getline(file, line)) {
        
        // --- WERYFIKACIJA $Entities ---
        if (line == "$Entities") {
            size_t numPoints, numCurves, numSurfaces, numVolumes;
            file >> numPoints >> numCurves >> numSurfaces >> numVolumes;
            
            size_t total_expected_entities = numPoints + numCurves + numSurfaces + numVolumes;
            ASSERT_EQ(mesh.entities.size(), total_expected_entities);

            size_t idx = 0;
            auto verify_entity = [&](int dim, size_t count) {
                for (size_t i = 0; i < count; ++i, ++idx) {
                    int tag;
                    file >> tag;
                    
                    double xmin, ymin, zmin, xmax, ymax, zmax;
                    if (dim == 0) {
                        file >> xmin >> ymin >> zmin;
                        xmax = xmin; ymax = ymin; zmax = zmin;
                    } else {
                        file >> xmin >> ymin >> zmin >> xmax >> ymax >> zmax;
                    }
                    
                    size_t numPhys;
                    file >> numPhys;
                    std::vector<int> file_phys_tags(numPhys);
                    for (size_t p = 0; p < numPhys; ++p) {
                        file >> file_phys_tags[p];
                    }

                    // Pomijamy granice topologiczne (tak jak dotychczas)
                    if (dim == 1) { size_t bp; file >> bp; for(size_t k=0; k<bp; ++k){int d; file>>d;} }
                    else if (dim == 2) { size_t bc; file >> bc; for(size_t k=0; k<bc; ++k){int d; file>>d;} }
                    else if (dim == 3) { size_t bs; file >> bs; for(size_t k=0; k<bs; ++k){int d; file>>d;} }

                    // Pełna weryfikacja encji
                    ASSERT_LT(idx, mesh.entities.size());
                    EXPECT_EQ(mesh.entities[idx].dim, dim);
                    EXPECT_EQ(mesh.entities[idx].tag, tag);
                    EXPECT_NEAR(mesh.entities[idx].bbox[0], xmin, 1e-4);
                    EXPECT_NEAR(mesh.entities[idx].bbox[1], ymin, 1e-4);
                    EXPECT_NEAR(mesh.entities[idx].bbox[2], zmin, 1e-4);
                    EXPECT_NEAR(mesh.entities[idx].bbox[3], xmax, 1e-4);
                    EXPECT_NEAR(mesh.entities[idx].bbox[4], ymax, 1e-4);
                    EXPECT_NEAR(mesh.entities[idx].bbox[5], zmax, 1e-4);
                    
                    // Sprawdzenie fizycznych tagów (jeśli Twój parser je zapisuje w MeshEntity)
                    EXPECT_EQ(mesh.entities[idx].physicalTags.size(), file_phys_tags.size());
                    for(size_t p = 0; p < file_phys_tags.size(); ++p) {
                        EXPECT_EQ(mesh.entities[idx].physicalTags[p], file_phys_tags[p]);
                    }
                }
            };

            verify_entity(0, numPoints);
            verify_entity(1, numCurves);
            verify_entity(2, numSurfaces);
            verify_entity(3, numVolumes);
        }
        
        // --- WERYFIKACIJA $Nodes ---
        else if (line == "$Nodes") {
            size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;
            file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;
            
            ASSERT_EQ(mesh.nodes.size(), numNodes);

            for (size_t b = 0; b < numEntityBlocks; ++b) {
                int entityDim, entityTag, parametric, numNodesInBlock;
                file >> entityDim >> entityTag >> parametric >> numNodesInBlock;

                std::vector<size_t> blockNodeTags(numNodesInBlock);
                for (int i = 0; i < numNodesInBlock; ++i) {
                    file >> blockNodeTags[i];
                }

                for (size_t i = 0; i < static_cast<size_t>(numNodesInBlock); ++i) {
                    double x, y, z;
                    file >> x >> y >> z;
                    
                    size_t global_idx = blockNodeTags[i] - 1;
                    ASSERT_LT(global_idx, mesh.nodes.size());
                    
                    EXPECT_NEAR(mesh.nodes[global_idx].x, static_cast<float>(x), 1e-5f);
                    EXPECT_NEAR(mesh.nodes[global_idx].y, static_cast<float>(y), 1e-5f);
                    EXPECT_NEAR(mesh.nodes[global_idx].z, static_cast<float>(z), 1e-5f);
                }
            }
        }
        
        // --- WERYFIKACIJA $Elements ---
        else if (line == "$Elements") {
            size_t numEntityBlocks, numElements, minElementTag, maxElementTag;
            file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;

            size_t mesh_elem_idx = 0;

            for (size_t b = 0; b < numEntityBlocks; ++b) {
                int entityDim, entityTag, elementType;
                size_t numElementsInBlock;
                file >> entityDim >> entityTag >> elementType >> numElementsInBlock;

                // Określamy liczbę węzłów dla danego typu elementu w Gmsh
                int nodes_per_elem = 0;
                if (elementType == 4) nodes_per_elem = 4;       // Tetraedr
                else if (elementType == 2) nodes_per_elem = 3;  // Trójkąt
                else if (elementType == 1) nodes_per_elem = 2;  // Linia
                else if (elementType == 15) nodes_per_elem = 1; // Punkt

                for (size_t i = 0; i < numElementsInBlock; ++i) {
                    size_t elemTag;
                    file >> elemTag;

                    std::vector<int> elem_nodes(nodes_per_elem);
                    for (int n = 0; n < nodes_per_elem; ++n) {
                        size_t nodeTag;
                        file >> nodeTag;
                        elem_nodes[n] = static_cast<int>(nodeTag) - 1;
                    }

                    // Jeśli to tetraedr (typ 4), to nasz parser go zapisał w mesh.elements
                    if (elementType == 4) {
                        ASSERT_LT(mesh_elem_idx, mesh.elements.size());
                        ASSERT_EQ(mesh.elements[mesh_elem_idx].size(), 4);
                        
                        for (int n = 0; n < 4; ++n) {
                            EXPECT_EQ(mesh.elements[mesh_elem_idx][n], elem_nodes[n]);
                        }
                        mesh_elem_idx++;
                    }
                }
            }
            // Sprawdzamy czy liczba wczytanych tetraedrów idealnie się zgadza
            EXPECT_EQ(mesh_elem_idx, mesh.elements.size());
        }
    }
    file.close();
    
    std::cout << "[Test Sukces] Test dynamiczny potwierdził pełną zgodność wczytanych danych z plikiem!\n";
}
