#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <memory>

#include "SimulationUtils.hpp"
#include "Mesh/Msh.hpp"



/// @brief Lekki widok pól na urządzeniu (przekazywany do kerneli SYCL)
struct FieldDeviceView {
    float* d_rho;          // Gęstość ładunku [num_nodes]
    Vector3* d_j;          // Prąd (x,y,z) [num_nodes]
    Vector3* d_E;          // Pole elektryczne (x,y,z) [num_nodes]
    Vector3* d_B;          // Pole magnetyczne X [num_nodes]
    size_t num_nodes;      // Liczba węzłów siatki
};

/// @brief Represents spatial grid fields (e.g., charge density, electric and magnetic fields) for the simulation.
class GridField
{
private:
    sycl::queue &queue_;
    std::unique_ptr<IMesh> mesh_;
    MeshDeviceView mesh_device_view_;
    FieldDeviceView field_device_view_;

public:
    explicit GridField(sycl::queue &q, const std::string& filename = "EMPTY") : queue_(q) {
        // Inicjalizacja pól wskaźnikowych jako nullptr na wypadek stanu "EMPTY"
        field_device_view_ = FieldDeviceView{nullptr, nullptr, nullptr, nullptr, 0};

        if(filename != "EMPTY") {
            std::filesystem::path filePath(filename);
            std::string ext = filePath.extension().string();
    
            if (ext == ".msh") {
                mesh_ = std::make_unique<Msh>(filename, queue_);
                mesh_device_view_ = mesh_->get_device_view();
                
                // Pobieramy liczbę węzłów siatki z widoku urządzenia
                field_device_view_.num_nodes = mesh_device_view_.get_total_nodes(); // Dostosuj do faktycznej metody w MeshDeviceView
                allocate_fields();
            } else {
                throw std::runtime_error("[GridField] Error: Unknown extension: " + ext);
            }
        }
    }

    ~GridField() { release_fields(); }

    /// @brief Blokada kopiowania (bezpieczeństwo zasobów VRAM)
    GridField(const GridField&) = delete;
    GridField& operator=(const GridField&) = delete;

    /// @brief Zwraca lekki widok pól na urządzenie do przekazania do kerneli SYCL
    FieldDeviceView get_field_device_view() const { return field_device_view_; }

    /// @brief Zwraca kolejkę SYCL przypisaną do siatki
    sycl::queue& get_queue() { return queue_; }

    /// @brief Zwraca lekki widok mesh na urządzeniu do przekazania do kernela SYCL
    MeshDeviceView get_mesh_device_view() const { return mesh_device_view_; }

    /// @brief Returns IMesh unique pointer
    const std::unique_ptr<IMesh>& get_mesh() const { return mesh_; }

private:
    /// @brief Allocates device memory for field_device_view_ pointers
    void allocate_fields() {
        size_t n = field_device_view_.num_nodes;
        if (n == 0) return;

        // Alokacja w pamięci USM Device
        field_device_view_.d_rho = sycl::malloc_device<float>(n, queue_);
        field_device_view_.d_j   = sycl::malloc_device<Vector3>(n, queue_);
        field_device_view_.d_E   = sycl::malloc_device<Vector3>(n, queue_);
        field_device_view_.d_B   = sycl::malloc_device<Vector3>(n, queue_);

        // Wyzerowanie pól na starcie
        queue_.fill(field_device_view_.d_rho, 0.0f, n);
        queue_.fill(field_device_view_.d_j,   Vector3{0,0,0}, n);
        queue_.fill(field_device_view_.d_E,   Vector3{0,0,0}, n);
        queue_.fill(field_device_view_.d_B,   Vector3{0,0,0}, n);
        queue_.wait();
    }

    /// @brief Frees all allocations from field_device_view_ (d_rho, d_j, d_E, d_B)
    void release_fields() {
        if (field_device_view_.d_rho) sycl::free(field_device_view_.d_rho, queue_);
        if (field_device_view_.d_j)   sycl::free(field_device_view_.d_j, queue_);
        if (field_device_view_.d_E)   sycl::free(field_device_view_.d_E, queue_);
        if (field_device_view_.d_B)   sycl::free(field_device_view_.d_B, queue_);
        
        field_device_view_ = FieldDeviceView{nullptr, nullptr, nullptr, nullptr, 0};
    }
};