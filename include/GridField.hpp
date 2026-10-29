#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <memory>

#include "SimulationUtils.hpp"
#include "Mesh/Msh.hpp"



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