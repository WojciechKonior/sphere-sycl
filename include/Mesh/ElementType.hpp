#pragma once
#include <array>
#include "Matrix.hpp"

using Vector3 = Matrix<float, 3, 1>;
using Vector3d = Matrix<double, 3, 1>;

using Matrix3x3 = Matrix<float, 3, 3>;

/**
 * @brief Represents a robust wrapper for mesh element types, matching Gmsh IDs.
 */
struct ElementType {
    /// @brief Underlying enumeration of supported element types and their standard Gmsh integer IDs.
    enum Value : int {
        Uninitialized    = 0, ///< Uninitialized or invalid element state.
        BoundaryTriangle = 2, ///< Boundary triangle element (Gmsh ID 2).
        BoundaryQuad     = 3, ///< Boundary quadrilateral element (Gmsh ID 3).
        Tetrahedron      = 4, ///< 3D tetrahedral element (Gmsh ID 4).
        Hexahedron       = 5, ///< 3D hexahedral element (Gmsh ID 5).
        Prism            = 6, ///< 3D wedge/prism element (Gmsh ID 6).
        Pyramid          = 7  ///< 3D pyramidal element (Gmsh ID 7).
    };

    Value val; ///< The active value of the element type.

    static constexpr size_t SIZE = 6;                                    ///< Total number of supported element types.
    static constexpr size_t FRONT = static_cast<size_t>(Value::BoundaryTriangle); ///< First valid raw element ID.
    static constexpr size_t BACK = FRONT + SIZE - 1;                     ///< Last valid raw element ID.

    /// @brief Compile-time array containing all valid element types for clean iteration.
    static constexpr std::array<Value, SIZE> ALL = []() constexpr {
        std::array<Value, SIZE> arr{};
        for (size_t i = 0; i < SIZE; ++i) {
            arr[i] = static_cast<Value>(FRONT + i);
        }
        return arr;
    }();

    /// @brief Default constructor initializing to Uninitialized state.
    constexpr ElementType() : val(Uninitialized) {}

    /// @brief Constructs an ElementType from the underlying enum Value.
    constexpr ElementType(Value v) : val(v) {}

    /// @brief Constructs an ElementType from a raw integer ID (e.g., from Gmsh).
    constexpr ElementType(int raw_id) : val(static_cast<Value>(raw_id)) {}

    /// @brief Implicit conversion operator to the underlying enum Value.
    constexpr operator Value() const { return val; }

    /// @brief Safely maps the element type to a zero-based array index [0, SIZE-1].
    /// @return The corresponding array index.
    /// @throws std::out_of_range if the element type is invalid or uninitialized.
    constexpr size_t to_index() const {
        int raw = static_cast<int>(val);
        if (raw < FRONT || raw > BACK) {
            throw std::out_of_range("Unknown or uninitialized ElementType!");
        }
        return static_cast<size_t>(raw - FRONT);
    }

    /// @brief Returns the number of nodes per element for the current type.
    /// @return Number of nodes (e.g., 8 for Hexahedron, 0 for Uninitialized).
    constexpr int get_num_nodes_per_element() const {
        switch (val) {
            case BoundaryTriangle: return 3;
            case BoundaryQuad:     return 4;
            case Tetrahedron:      return 4;
            case Hexahedron:       return 8;
            case Prism:            return 6;
            case Pyramid:          return 5;
            default: return 0;
        }
    }

    /// @brief Computes the element volume on the device using node coordinates.
    inline float device_compute_element_volume(const Vector3* d_nodes, const int* element_indices) const {
        auto tet_vol = [&](int a, int b, int c, int d) {
            auto r0 = static_cast<Vector3d>(d_nodes[element_indices[a]]);
            auto r1 = static_cast<Vector3d>(d_nodes[element_indices[b]]);
            auto r2 = static_cast<Vector3d>(d_nodes[element_indices[c]]);
            auto r3 = static_cast<Vector3d>(d_nodes[element_indices[d]]);
            auto v1 = Vector3d::sub(r1, r0);
            auto v2 = Vector3d::sub(r2, r0);
            auto v3 = Vector3d::sub(r3, r0);
            double det = Vector3d::dot(v1, Vector3d::cross(v2, v3));
            return sycl::fabs(det) / 6.0;
        };

        auto tri_area = [&](int a, int b, int c) {
            auto r0 = static_cast<Vector3d>(d_nodes[element_indices[a]]);
            auto r1 = static_cast<Vector3d>(d_nodes[element_indices[b]]);
            auto r2 = static_cast<Vector3d>(d_nodes[element_indices[c]]);
            auto v1 = Vector3d::sub(r1, r0);
            auto v2 = Vector3d::sub(r2, r0);
            auto cross_v = Vector3d::cross(v1, v2);
            double double_area_sq = cross_v.x * cross_v.x + cross_v.y * cross_v.y + cross_v.z * cross_v.z;
            return 0.5 * sycl::sqrt(double_area_sq);
        };

        switch (val) {
            case BoundaryTriangle:
                return static_cast<float>(tri_area(0, 1, 2));
            case BoundaryQuad:
                return static_cast<float>(tri_area(0, 1, 2) + tri_area(0, 2, 3));
            case Tetrahedron:
                return static_cast<float>(tet_vol(0, 1, 2, 3));
            case Pyramid:
                return static_cast<float>(tet_vol(0, 1, 3, 4) + tet_vol(1, 2, 3, 4));
            case Prism:
                return static_cast<float>(tet_vol(0, 1, 2, 3) + tet_vol(1, 4, 2, 3) + tet_vol(2, 4, 5, 3));
            case Hexahedron:
                return static_cast<float>(tet_vol(0, 1, 3, 4) + tet_vol(1, 2, 3, 6) + tet_vol(4, 5, 6, 1) + tet_vol(4, 6, 7, 3) + tet_vol(1, 3, 4, 6));
            default:
                return 0.0f;
        }
    }

    /// @brief Computes the element centroid on the device using node coordinates.
    inline Vector3 device_compute_element_centroid(const Vector3* d_nodes, const int* element_indices) const {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;

        for (size_t i = 0; i < get_num_nodes_per_element(); ++i) {
            auto node = static_cast<Vector3d>(d_nodes[element_indices[i]]);
            sum_x += node.x;
            sum_y += node.y;
            sum_z += node.z;
        }

        double inv_n = 1.0 / static_cast<double>(get_num_nodes_per_element());
        return Vector3{
            static_cast<float>(sum_x * inv_n),
            static_cast<float>(sum_y * inv_n),
            static_cast<float>(sum_z * inv_n)
        };
    }

    /// @brief Oblicza macierz Jakobianu 3x3 dla dowolnego typu elementu na urządzeniu (GPU).
    /// @param d_nodes Wskaźnik na tablicę wszystkich węzłów siatki.
    /// @param element_indices Wskaźnik na indeksy węzłów bieżącego elementu.
    /// @return Macierz Jakobianu [3x3].
    inline Matrix3x3 device_compute_element_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        switch (val) {
            case BoundaryTriangle:
                return device_compute_triangle_jacobian(d_nodes, element_indices);
            case BoundaryQuad:
                return device_compute_quad_jacobian(d_nodes, element_indices);
            case Tetrahedron:
                return device_compute_tetrahedron_jacobian(d_nodes, element_indices);
            case Pyramid:
                return device_compute_pyramid_jacobian(d_nodes, element_indices);
            case Prism:
                return device_compute_prism_jacobian(d_nodes, element_indices);
            case Hexahedron:
                return device_compute_hexahedron_jacobian(d_nodes, element_indices);
            default:
                return Matrix3x3();
        }
    }

    /// @brief Computes 3x3 Jacobian matrix for triangle element on GPU.
    inline Matrix3x3 device_compute_triangle_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        // 1. Pobieramy 3 węzły trójkąta brzegowego
        Vector3d r[] = { get_node(0), get_node(1), get_node(2) };

        // 2. Kolumny Jakobianu 3x2 (wektory krawędziowe / styczne do powierzchni)
        // c0 = ∂p/∂ξ = p1 - p0
        // c1 = ∂p/∂η = p2 - p0
        Vector3d c0 = Vector3d::sub(r[1], r[0]);
        Vector3d c1 = Vector3d::sub(r[2], r[0]);

        // 3. Trzecia kolumna jako wektor normalny (do domknięcia do macierzy 3x3)
        Vector3d c2 = Vector3d::cross(c0, c1);

        // 4. Budowa finalnej macierzy 3x3
        return static_cast<Matrix3x3>(Vector3d::hmerge(c0, c1, c2));
    }

    /// @brief Computes 3x3 Jacobian matrix for quad element on GPU.
    inline Matrix3x3 device_compute_quad_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        // 1. Pobieramy 4 węzły czworokąta brzegowego (zgodnie z kolejnością GMSH)
        Vector3d r[] = { get_node(0), get_node(1), get_node(2), get_node(3) };

        // 2. Kolumny Jakobianu 3x2 w środku elementu (xi=0, eta=0) jako średnie krawędzi przeciwległych
        // c0 = ∂p/∂ξ = 0.5 * [(p1 - p0) + (p2 - p3)]
        // c1 = ∂p/∂η = 0.5 * [(p3 - p0) + (p2 - p1)]
        Vector3d edge01 = Vector3d::sub(r[1], r[0]);
        Vector3d edge32 = Vector3d::sub(r[2], r[3]);
        Vector3d c0 = {
            0.5 * (edge01.x + edge32.x),
            0.5 * (edge01.y + edge32.y),
            0.5 * (edge01.z + edge32.z)
        };

        Vector3d edge03 = Vector3d::sub(r[3], r[0]);
        Vector3d edge12 = Vector3d::sub(r[2], r[1]);
        Vector3d c1 = {
            0.5 * (edge03.x + edge12.x),
            0.5 * (edge03.y + edge12.y),
            0.5 * (edge03.z + edge12.z)
        };

        // 3. Trzecia kolumna jako wektor normalny do powierzchni
        Vector3d c2 = Vector3d::cross(c0, c1);

        // 4. Budowa finalnej macierzy 3x3
        return static_cast<Matrix3x3>(Vector3d::hmerge(c0, c1, c2));
    }

    /// @brief Computes 3x3 Jacobian matrix for tetrahedron element on GPU.
    inline Matrix3x3 device_compute_tetrahedron_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        // 1. Pobieramy 4 węzły czworościanu
        Vector3d r[] = { get_node(0), get_node(1), get_node(2), get_node(3) };

        // 2. Kolumny Jakobianu 3x3 jako wektory krawędziowe wychodzące z węzła 0
        // c0 = ∂p/∂ξ = p1 - p0
        // c1 = ∂p/∂η = p2 - p0
        // c2 = ∂p/∂ζ = p3 - p0
        Vector3d c0 = Vector3d::sub(r[1], r[0]);
        Vector3d c1 = Vector3d::sub(r[2], r[0]);
        Vector3d c2 = Vector3d::sub(r[3], r[0]);

        // 3. Budowa finalnej macierzy 3x3 za pomocą hmerge
        return static_cast<Matrix3x3>(Vector3d::hmerge(c0, c1, c2));
    }

    /// @brief Computes 3x3 Jacobian matrix for pyramid element on GPU.
    inline Matrix3x3 device_compute_pyramid_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        // 1. Pobieramy 5 węzłów piramidy (0-3 podstawa, 4 szczyt)
        Vector3d r[] = { get_node(0), get_node(1), get_node(2), get_node(3), get_node(4) };

        // 2. Kolumny Jakobianu 3x3 w środku elementu (xi=0, eta=0, zeta=0)
        // c0 = 0.25 * (-p0 + p1 + p2 - p3)
        // c1 = 0.25 * (-p0 - p1 + p2 + p3)
        // c2 = p4 - 0.25 * (p0 + p1 + p2 + p3)  (wektor od środka podstawy do szczytu)
        
        auto sum_base = Vector3d::add(
            Vector3d::add(r[0], r[1]),
            Vector3d::add(r[2], r[3])
        );
        auto base_center = Vector3d::scale(sum_base, 0.25);

        Vector3d c0 = {
            0.25 * (-r[0].x + r[1].x + r[2].x - r[3].x),
            0.25 * (-r[0].y + r[1].y + r[2].y - r[3].y),
            0.25 * (-r[0].z + r[1].z + r[2].z - r[3].z)
        };

        Vector3d c1 = {
            0.25 * (-r[0].x - r[1].x + r[2].x + r[3].x),
            0.25 * (-r[0].y - r[1].y + r[2].y + r[3].y),
            0.25 * (-r[0].z - r[1].z + r[2].z + r[3].z)
        };

        Vector3d c2 = Vector3d::sub(r[4], base_center);

        // 3. Scalenie za pomocą hmerge i konwersja do float 3x3
        return static_cast<Matrix3x3>(Vector3d::hmerge(c0, c1, c2));
    }

    /// @brief Computes 3x3 Jacobian matrix for prism element on GPU.
    inline Matrix3x3 device_compute_prism_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        // 1. Pobieramy 6 węzłów graniastosłupa (0-2 dół, 3-5 góra)
        Vector3d r[] = { 
            get_node(0), get_node(1), get_node(2), 
            get_node(3), get_node(4), get_node(5) 
        };

        // 2. Kolumny Jakobianu 3x3 w środku elementu
        // c0 = 0.5 * [(p1 - p0) + (p4 - p3)]
        // c1 = 0.5 * [(p2 - p0) + (p5 - p3)]
        // c2 = (1/6) * [(p3 - p0) + (p4 - p1) + (p5 - p2)]
        
        auto edge01 = Vector3d::sub(r[1], r[0]);
        auto edge34 = Vector3d::sub(r[4], r[3]);
        Vector3d c0 = Vector3d::scale(
            Vector3d::add(edge01, edge34), 0.5
        );

        auto edge02 = Vector3d::sub(r[2], r[0]);
        auto edge35 = Vector3d::sub(r[5], r[3]);
        Vector3d c1 = Vector3d::scale(
            Vector3d::add(edge02, edge35), 0.5
        );

        auto vert03 = Vector3d::sub(r[3], r[0]);
        auto vert14 = Vector3d::sub(r[4], r[1]);
        auto vert25 = Vector3d::sub(r[5], r[2]);
        
        auto sum_verts = Vector3d::add(
            Vector3d::add(vert03, vert14), 
            vert25
        );
        Vector3d c2 = Vector3d::scale(sum_verts, 1.0 / 6.0);

        // 3. Scalenie za pomocą hmerge i konwersja do float 3x3
        return static_cast<Matrix3x3>(Vector3d::hmerge(c0, c1, c2));
    }

    /// @brief Computes 3x3 Jacobian matrix for hexahedron element on GPU.
    inline Matrix3x3 device_compute_hexahedron_jacobian(const Vector3* d_nodes, const int* element_indices) const {
        auto get_node = [&](int idx) { return static_cast<Vector3d>(d_nodes[element_indices[idx]]); };
        // 1. Pobieramy 8 węzłów sześcianu (0-3 dół, 4-7 góra)
        Vector3d r[] = { 
            get_node(0), get_node(1), get_node(2), get_node(3), 
            get_node(4), get_node(5), get_node(6), get_node(7) 
        };

        // 2. Kolumny Jakobianu 3x3 w środku elementu jako średnie 4 krawędzi
        auto c0 = Vector3d::scale(
            Vector3d::add(
                Vector3d::add(Vector3d::sub(r[1], r[0]), Vector3d::sub(r[2], r[3])),
                Vector3d::add(Vector3d::sub(r[5], r[4]), Vector3d::sub(r[6], r[7]))
            ), 0.125
        );

        auto c1 = Vector3d::scale(
            Vector3d::add(
                Vector3d::add(Vector3d::sub(r[3], r[0]), Vector3d::sub(r[2], r[1])),
                Vector3d::add(Vector3d::sub(r[7], r[4]), Vector3d::sub(r[6], r[5]))
            ), 0.125
        );

        auto c2 = Vector3d::scale(
            Vector3d::add(
                Vector3d::add(Vector3d::sub(r[4], r[0]), Vector3d::sub(r[5], r[1])),
                Vector3d::add(Vector3d::sub(r[6], r[2]), Vector3d::sub(r[7], r[3]))
            ), 0.125
        );

        // 3. Scalenie za pomocą hmerge i konwersja do float 3x3
        return static_cast<Matrix3x3>(Vector3d::hmerge(c0, c1, c2));
    }
};