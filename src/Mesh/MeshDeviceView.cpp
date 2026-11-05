#include "Mesh/MeshDeviceView.hpp"

ElementsDeviceView &MeshDeviceView::elements_at(ElementType type)
{
    return d_elements[type.to_index()];
}

const ElementsDeviceView &MeshDeviceView::elements_at(ElementType type) const
{
    return d_elements[type.to_index()];
}

ElementsDeviceView &MeshDeviceView::elements_at(int raw_gmsh_id)
{
    return d_elements[ElementType(raw_gmsh_id).to_index()];
}

const ElementsDeviceView &MeshDeviceView::elements_at(int raw_gmsh_id) const
{
    return d_elements[ElementType(raw_gmsh_id).to_index()];
}

void MeshDeviceView::compute_element_volumes(sycl::queue &q)
{
    for (ElementType type : ElementType::ALL)
    {
        auto &d_elements = elements_at(type);

        // Jeśli dany typ elementów nie występuje w siatce, pomijamy
        if (d_elements.num_indices == 0 || d_elements.nodes_per_element == 0)
        {
            d_elements.volumes = nullptr;
            continue;
        }

        size_t num_elements = d_elements.num_elements;

        // 1. Alokacja pamięci USM na urządzeniu dla objętości elementów
        float *d_volumes_mutable = sycl::malloc_device<float>(num_elements, q);
        d_elements.volumes = d_volumes_mutable; // Przypisanie doconst wskaźnika w widoku

        // Pobieramy stałe widoki do przekazania przez wartość do lambdy SYCL
        const Vector3 *d_nodes_ptr = d_nodes.nodes;
        const int *d_indices_ptr = d_elements.indices;
        size_t nodes_per_elem = d_elements.nodes_per_element;
        ElementType current_type = type;

        // 2. Uruchomienie kernela SYCL za pomocą parallel_for
        q.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1> idx)
        {
            size_t e = idx[0];
            size_t offset = e * nodes_per_elem;
            const int* element_indices = d_indices_ptr + offset;
            d_volumes_mutable[e] = current_type.device_compute_element_volume(d_nodes_ptr, element_indices); 
        }).wait(); // Czekamy na zakończenie obliczeń kernela
    }
}

void MeshDeviceView::compute_element_centroids(sycl::queue &q)
{
    for (ElementType type : ElementType::ALL)
    {
        auto &d_elements = elements_at(type);

        if (d_elements.num_indices == 0 || d_elements.nodes_per_element == 0)
        {
            d_elements.centroids = nullptr;
            continue;
        }

        size_t num_elements = d_elements.num_elements;

        // 1. Alokacja pamięci USM na urządzeniu dla środków ciężkości
        Vector3 *d_centroids_mutable = sycl::malloc_device<Vector3>(num_elements, q);
        d_elements.centroids = d_centroids_mutable;

        const Vector3 *d_nodes_ptr = d_nodes.nodes;
        const int *d_indices_ptr = d_elements.indices;
        size_t nodes_per_elem = d_elements.nodes_per_element;
        ElementType current_type = type;

        // 2. Uruchomienie kernela SYCL
        q.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1> idx)
        {
            size_t e = idx[0];
            size_t offset = e * nodes_per_elem;
            const int* element_indices = d_indices_ptr + offset;
            d_centroids_mutable[e] = current_type.device_compute_element_centroid(d_nodes_ptr, element_indices); 
        }).wait();
    }
}

void MeshDeviceView::compute_element_jacobians(sycl::queue &q)
{
    for (ElementType type : ElementType::ALL)
    {
        auto &d_elements = elements_at(type);

        // Jeśli element nie wspiera Jakobianu lub nie występuje w siatce
        if (d_elements.num_indices == 0 || d_elements.nodes_per_element == 0)
        {
            d_elements.jacobians = nullptr;
            continue;
        }

        size_t num_elements = d_elements.num_elements;

        // 1. Alokacja pamięci USM na urządzeniu dla macierzy Jakobianu (3x3 dla przestrzeni 3D)
        Matrix<float, 3, 3> *d_jacobians_mutable = sycl::malloc_device<Matrix<float, 3, 3>>(num_elements, q);
        d_elements.jacobians = d_jacobians_mutable;

        const Vector3 *d_nodes_ptr = d_nodes.nodes;
        const int *d_indices_ptr = d_elements.indices;
        size_t nodes_per_elem = d_elements.nodes_per_element;
        ElementType current_type = type;

        // 2. Uruchomienie kernela SYCL do równoległego liczenia Jakobianów
        q.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1> idx)
        {
            size_t e = idx[0];
            size_t offset = e * nodes_per_elem;
            const int* element_indices = d_indices_ptr + offset;
            
            // Założenie: ElementType posiada metodę device_compute_element_jacobian 
            // przyjmującą węzły oraz wskaźnik na indeksy lokalne elementu.
            // Zwraca ona obiekt typu Matrix<float, 3, 3>.
            d_jacobians_mutable[e] = current_type.device_compute_element_jacobian(d_nodes_ptr, element_indices); 
        }).wait();
    }
}