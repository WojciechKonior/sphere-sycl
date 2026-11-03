#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <filesystem>
#include <vector>
#include <map>
#include "Mesh/Msh.hpp"

/**
 * @class GridFieldTest
 * @brief Test fixture for GridFieldTest class unit tests.
 */
class MshTest : public ::testing::Test {
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
TEST_F(MshTest, DynamicFileParityVerificationForHexahedronAndTetrahedronMsh) {
    std::vector<std::string> filepaths = {std::string(TEST_ASSETS_DIR) + "/simple_box_tet.msh", std::string(TEST_ASSETS_DIR) + "/simple_box_hex.msh"};
    for(const auto& filepath : filepaths){
        ASSERT_TRUE(std::filesystem::exists(filepath)) 
            << "Brak pliku siatki pod ścieżką: " << std::filesystem::absolute(filepath);
    
        // 1. Wczytujemy siatkę za pomocą naszej klasy
        Msh mesh(filepath, queue);
    
        // 2. Otwieramy ten sam plik bezpośrednio w teście do weryfikacji na żywo
        std::ifstream file(filepath);
        ASSERT_TRUE(file.is_open());
    
        std::string line;
        while (std::getline(file, line)) {
            
            // --- WERYFIKACIJA $Entities ---
            if (line.find("$Entities") != std::string::npos) {
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
            else if (line.find("$Nodes") != std::string::npos) {
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
    
                    int num_params_per_node = (parametric == 1) ? entityDim : (parametric == 2 ? 2 * entityDim : 0);
    
                    for (size_t i = 0; i < static_cast<size_t>(numNodesInBlock); ++i) {
                        double x, y, z;
                        file >> x >> y >> z;
    
                        for (int p = 0; p < num_params_per_node; ++p) {
                            double dummy;
                            file >> dummy;
                        }
                        
                        if (blockNodeTags[i] == 0) continue; // Zabezpieczenie przed podmiarowym tagiem 0
    
                        size_t global_idx = blockNodeTags[i] - 1;
                        ASSERT_LT(global_idx, mesh.nodes.size());
                        
                        EXPECT_NEAR(mesh.nodes[global_idx].x, static_cast<float>(x), 1e-4f);
                        EXPECT_NEAR(mesh.nodes[global_idx].y, static_cast<float>(y), 1e-4f);
                        EXPECT_NEAR(mesh.nodes[global_idx].z, static_cast<float>(z), 1e-4f);
                    }
                }
            }
            
            // --- WERYFIKACIJA $Elements ---
            else if (line.find("$Elements") != std::string::npos) {
                size_t numEntityBlocks, numElements, minElementTag, maxElementTag;
                file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;
    
                // Słownik przechowujący elementy wczytane bezpośrednio z pliku, pogrupowane po elementType
                std::map<int, std::vector<std::vector<int>>> file_elements_by_type;
    
                for (size_t b = 0; b < numEntityBlocks; ++b) {
                    int entityDim, entityTag, elementType;
                    size_t numElementsInBlock;
                    file >> entityDim >> entityTag >> elementType >> numElementsInBlock;
    
                    // Określamy liczbę węzłów na podstawie typu Gmsh
                    int nodes_per_elem = 0;
                    if      (elementType == 15) nodes_per_elem = 1; // Punkt
                    else if (elementType == 1)  nodes_per_elem = 2; // Linia
                    else if (elementType == 2)  nodes_per_elem = 3; // Trójkąt
                    else if (elementType == 3)  nodes_per_elem = 4; // Czworokąt
                    else if (elementType == 4)  nodes_per_elem = 4; // Tetraedr
                    else if (elementType == 5)  nodes_per_elem = 8; // Heksaedr
                    else if (elementType == 6)  nodes_per_elem = 6; // Pryzma
                    else if (elementType == 7)  nodes_per_elem = 5; // Piramida
                    else {
                        throw std::runtime_error("[Test Error] Nieobsługiwany elementType w teście: " + std::to_string(elementType));
                    }
    
                    for (size_t i = 0; i < numElementsInBlock; ++i) {
                        size_t elemTag;
                        file >> elemTag;
    
                        std::vector<int> elem_nodes(nodes_per_elem);
                        for (int n = 0; n < nodes_per_elem; ++n) {
                            size_t nodeTag;
                            file >> nodeTag;
                            elem_nodes[n] = static_cast<int>(nodeTag) - 1;
                        }
    
                        file_elements_by_type[elementType].push_back(elem_nodes);
                    }
                }
    
                // Lambda do weryfikacji dowolnej grupy elementów w elements_
                auto verify_group = [](const std::vector<std::vector<int>>& file_list,
                                         const std::vector<int>& mesh_vector,
                                         int stride,
                                         const std::string& type_name) 
                {
                    ASSERT_EQ(file_list.size(), mesh_vector.size() / stride) 
                        << "Niezgodność liczby elementów dla typu: " << type_name;
    
                    for (const auto& expected_nodes : file_list) {
                        bool found = false;
                        size_t total_elements = mesh_vector.size() / stride;
    
                        for (size_t t = 0; t < total_elements; ++t) {
                            size_t offset = t * stride;
                            bool match = true;
                            for (int n = 0; n < stride; ++n) {
                                if (mesh_vector[offset + n] != expected_nodes[n]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                found = true;
                                break;
                            }
                        }
    
                        EXPECT_TRUE(found) << "Błąd: Element typu [" << type_name  << "] nie został odnaleziony w strukturze elements!";
                    }
                };
    
                // Weryfikujemy każdą wykrytą grupę w strukturze elements_
                if (file_elements_by_type.count(4)) {
                    verify_group(file_elements_by_type[4], mesh.elements.at(ElementType::Tetrahedron), 4, "Tetrahedron");
                }
                if (file_elements_by_type.count(5)) {
                    verify_group(file_elements_by_type[5], mesh.elements.at(ElementType::Hexahedron), 8, "Hexahedron");
                }
                if (file_elements_by_type.count(6)) {
                    verify_group(file_elements_by_type[6], mesh.elements.at(ElementType::Prism), 6, "Prism");
                }
                if (file_elements_by_type.count(7)) {
                    verify_group(file_elements_by_type[7], mesh.elements.at(ElementType::Pyramid), 5, "Pyramid");
                }
                if (file_elements_by_type.count(2)) {
                    verify_group(file_elements_by_type[2], mesh.elements.at(ElementType::BoundaryTriangle), 3, "Boundary Triangle");
                }
                if (file_elements_by_type.count(3)) {
                    verify_group(file_elements_by_type[3], mesh.elements.at(ElementType::BoundaryQuad), 4, "Boundary Quadrangle");
                }
            }
        }
        file.close();
        
        std::cout << "[Test Sukces] Test dynamiczny potwierdził pełną zgodność wczytanych danych z plikiem!\n";
    }
    std::string filepath = std::string(TEST_ASSETS_DIR) + "/simple_box_hex.msh";
    
}

/**
 * @brief Verifies that MeshDeviceView can be successfully passed into a SYCL kernel 
 * and accessed on the device without throwing exceptions or causing memory faults.
 */
TEST_F(MshTest, MeshDeviceViewSyclKernelExecutionTest) {
    std::string filepath = std::string(TEST_ASSETS_DIR) + "/simple_box_tet.msh";
    ASSERT_TRUE(std::filesystem::exists(filepath)) << "Brak pliku siatki: " << filepath;

    // 1. Wczytujemy siatkę (co automatycznie inicjalizuje i kopiuje dane do mesh_device_view)
    Msh mesh(filepath, queue);

    // Kopiujemy widok urządzenia przez wartość (jest bezpieczny dla device/host)
    MeshDeviceView dev_view = mesh.get_device_view(); // Załóżmy, że maszgetter lub pole jest publiczne

    // 2. Odpalamy małe zadanie w SYCL, żeby przetestować dostęp do struktur
    try {
        queue.submit([&](sycl::handler &cgh) {
            cgh.single_task([=]() {
                // Iterujemy po tablicy d_elements wewnątrz kernela SYCL
                for (size_t i = 0; i < 6; ++i) {
                    const auto& elem_view = dev_view.d_elements[i];
                    
                    // Bezpieczny odczyt na urządzeniu (jeśli wskaźnik nie jest nullptr)
                    if (elem_view.indices != nullptr && elem_view.num_indices > 0) {
                        // Możemy odczytać np. pierwszy indeks elementu jako test spójności
                        int first_node = elem_view.indices[0];
                        (void)first_node; // Unikamy ostrzeżenia o nieużywanej zmiennej
                    }
                }
            });
        });
        
        // Czekamy na zakończenie wykonania kernela
        queue.wait();
    } catch (const sycl::exception &e) {
        FAIL() << "SYCL kernel execution failed: " << e.what();
    }

    SUCCEED() << "MeshDeviceView został pomyślnie przekazany i odczytany w kenerlu SYCL!";
}

/**
 * @test Verifies that mesh nodes and elements are correctly uploaded to device memory 
 * and match the raw .msh file content via MeshDeviceView.
 */
TEST_F(MshTest, DynamicDeviceViewParityVerification) {
    std::vector<std::string> filepaths = {
        std::string(TEST_ASSETS_DIR) + "/simple_box_tet.msh", 
        std::string(TEST_ASSETS_DIR) + "/simple_box_hex.msh"
    };

    for (const auto& filepath : filepaths) {
        ASSERT_TRUE(std::filesystem::exists(filepath)) 
            << "Brak pliku siatki pod ścieżką: " << std::filesystem::absolute(filepath);
    
        // 1. Wczytujemy siatkę (co alokuje pamięć USM i kopiuje dane na GPU)
        Msh mesh(filepath, queue);
    
        // Pobieramy widok urządzenia z obiektu Msh (zakładając metodę get_device_view() lub publiczne pole)
        MeshDeviceView dev_view = mesh.get_device_view(); 

        // 2. Otwieramy plik bezpośrednio w teście do weryfikacji danych na żywo
        std::ifstream file(filepath);
        ASSERT_TRUE(file.is_open());
    
        std::string line;
        while (std::getline(file, line)) {
            
            // --- WERYFIKACIJA $Nodes NA URZĄDZENIU ---
            if (line.find("$Nodes") != std::string::npos) {
                size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;
                file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;
                
                ASSERT_EQ(dev_view.num_nodes, numNodes);

                // Kopiujemy węzły z pamięci urządzeniowej (USM) z powrotem na hosta do weryfikacji
                std::vector<Vector3> h_nodes_from_device(numNodes);
                queue.copy(dev_view.d_nodes, h_nodes_from_device.data(), numNodes).wait();
    
                for (size_t b = 0; b < numEntityBlocks; ++b) {
                    int entityDim, entityTag, parametric, numNodesInBlock;
                    file >> entityDim >> entityTag >> parametric >> numNodesInBlock;
    
                    std::vector<size_t> blockNodeTags(numNodesInBlock);
                    for (int i = 0; i < numNodesInBlock; ++i) {
                        file >> blockNodeTags[i];
                    }
    
                    int num_params_per_node = (parametric == 1) ? entityDim : (parametric == 2 ? 2 * entityDim : 0);
    
                    for (size_t i = 0; i < static_cast<size_t>(numNodesInBlock); ++i) {
                        double x, y, z;
                        file >> x >> y >> z;
    
                        for (int p = 0; p < num_params_per_node; ++p) {
                            double dummy;
                            file >> dummy;
                        }
                        
                        if (blockNodeTags[i] == 0) continue;
    
                        size_t global_idx = blockNodeTags[i] - 1;
                        ASSERT_LT(global_idx, numNodes);
                        
                        // Sprawdzamy współrzędne pobrane bezpośrednio z bufora GPU (`d_nodes`)
                        EXPECT_NEAR(h_nodes_from_device[global_idx].x, static_cast<float>(x), 1e-4f);
                        EXPECT_NEAR(h_nodes_from_device[global_idx].y, static_cast<float>(y), 1e-4f);
                        EXPECT_NEAR(h_nodes_from_device[global_idx].z, static_cast<float>(z), 1e-4f);
                    }
                }
            }
            
            // --- WERYFIKACIJA $Elements NA URZĄDZENIU ---
            else if (line.find("$Elements") != std::string::npos) {
                size_t numEntityBlocks, numElements, minElementTag, maxElementTag;
                file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;
    
                std::map<int, std::vector<std::vector<int>>> file_elements_by_type;
    
                for (size_t b = 0; b < numEntityBlocks; ++b) {
                    int entityDim, entityTag, elementType;
                    size_t numElementsInBlock;
                    file >> entityDim >> entityTag >> elementType >> numElementsInBlock;
    
                    int nodes_per_elem = 0;
                    if      (elementType == 15) nodes_per_elem = 1;
                    else if (elementType == 1)  nodes_per_elem = 2;
                    else if (elementType == 2)  nodes_per_elem = 3;
                    else if (elementType == 3)  nodes_per_elem = 4;
                    else if (elementType == 4)  nodes_per_elem = 4;
                    else if (elementType == 5)  nodes_per_elem = 8;
                    else if (elementType == 6)  nodes_per_elem = 6;
                    else if (elementType == 7)  nodes_per_elem = 5;
                    else {
                        throw std::runtime_error("[Test Error] Nieobsługiwany elementType: " + std::to_string(elementType));
                    }
    
                    for (size_t i = 0; i < numElementsInBlock; ++i) {
                        size_t elemTag;
                        file >> elemTag;
    
                        std::vector<int> elem_nodes(nodes_per_elem);
                        for (int n = 0; n < nodes_per_elem; ++n) {
                            size_t nodeTag;
                            file >> nodeTag;
                            elem_nodes[n] = static_cast<int>(nodeTag) - 1;
                        }
    
                        file_elements_by_type[elementType].push_back(elem_nodes);
                    }
                }
    
                // Lambda weryfikująca grupy elementów pobrane z `MeshDeviceView::elements_at`
                auto verify_device_group = [&](int gmsh_type_id, int stride, const std::string& type_name) {
                    if (!file_elements_by_type.count(gmsh_type_id)) return;

                    const auto& file_list = file_elements_by_type[gmsh_type_id];
                    
                    // Pobieramy widok elementów z GPU dla danego typu
                    const auto& dev_elem_view = dev_view.elements_at(gmsh_type_id);

                    ASSERT_EQ(dev_elem_view.nodes_per_element, static_cast<size_t>(stride));

                    // Jeśli w pliku są elementy tego typu, bufor na GPU musi istnieć
                    if (!file_list.empty()) {
                        ASSERT_NE(dev_elem_view.indices, nullptr);
                        ASSERT_EQ(dev_elem_view.num_indices, file_list.size() * stride);

                        // Kopiujemy wskaźnik indeksów z GPU na hosta w celu weryfikacji
                        std::vector<int> h_indices_from_device(dev_elem_view.num_indices);
                        queue.copy(dev_elem_view.indices, h_indices_from_device.data(), dev_elem_view.num_indices).wait();

                        for (const auto& expected_nodes : file_list) {
                            bool found = false;
                            size_t total_elements = dev_elem_view.num_indices / stride;

                            for (size_t t = 0; t < total_elements; ++t) {
                                size_t offset = t * stride;
                                bool match = true;
                                for (int n = 0; n < stride; ++n) {
                                    if (h_indices_from_device[offset + n] != expected_nodes[n]) {
                                        match = false;
                                        break;
                                    }
                                }
                                if (match) {
                                    found = true;
                                    break;
                                }
                            }
                            EXPECT_TRUE(found) << "Błąd GPU: Element typu [" << type_name << "] nie został odnaleziony w pamięci urządzenia!";
                        }
                    } else {
                        // Jeśli brak elementów w pliku, wskaźnik powinien być nullptr
                        EXPECT_EQ(dev_elem_view.indices, nullptr);
                        EXPECT_EQ(dev_elem_view.num_indices, 0);
                    }
                };

                // Weryfikacja poszczególnych typów za pomocą nowego systemu `elements_at` na urządzeniu
                verify_device_group(4, 4, "Tetrahedron");
                verify_device_group(5, 8, "Hexahedron");
                verify_device_group(6, 6, "Prism");
                verify_device_group(7, 5, "Pyramid");
                verify_device_group(2, 3, "Boundary Triangle");
                verify_device_group(3, 4, "Boundary Quadrangle");
            }
        }
        file.close();
        
        std::cout << "[Test Sukces] Weryfikacja widoku urządzenia (GPU) zakończona pomyślnie dla: " << filepath << "\n";
    }
}