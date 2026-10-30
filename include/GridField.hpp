#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <memory>

#include "SimulationUtils.hpp"
#include "Mesh/Msh.hpp"

/// @brief Represents spatial grid fields (e.g., charge density, electric and magnetic fields) for the simulation.
class GridField
{
private:
    sycl::queue &queue_;
    std::unique_ptr<IMesh> mesh_;
    MeshDeviceView mesh_device_view_;

public:
    explicit GridField(sycl::queue &q, const std::string& filename = "EMPTY") : queue_(q)
        {
            if(filename != "EMPTY")
            {
                std::filesystem::path filePath(filename);
                std::string ext = filePath.extension().string();
    
                // Obsługa rozszerzenia .msh
                if (ext == ".msh") {
                    mesh_ = std::make_unique<Msh>(filename, queue_);
                    // mesh_device_view_ = mesh_->get
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
    MeshDeviceView get_device_view() const { return mesh_device_view_; }

    const std::unique_ptr<IMesh>& get_mesh() const { return mesh_; }
};