#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <concepts>
#include "Species.hpp"

class ParticleStorage;
class GridField;

namespace SimulationUtils {
    /// @brief Physical and simulation configuration constants.
    namespace Constants {
        // Simulation constants
        constexpr double MACROPARTICLE_WEIGHT    = 1.0e8f;
        constexpr size_t CHUNK_CAPACITY         = 2000000; // 2e6 -> ((6*4 + 4/8)*2e6 + 8 + 8 + 4)B = 46,73MB per chunk
    
        // physical constants
        constexpr double MASS_ELECTRON           = 9.109e-31f;
        constexpr double MASS_HYDROGEN           = 1.674e-27f;
        constexpr double MASS_HYDROGEN_ION       = 1.672e-27f;
        constexpr double CHARGE_ELEMENTARY       = 1.602e-19f;
        constexpr double CHARGE_ELECTRON         = -CHARGE_ELEMENTARY;
        constexpr double CHARGE_HYDROGEN_ION     =  CHARGE_ELEMENTARY;
        constexpr double CHARGE_NEUTRAL          =  0.0f;
        constexpr double k_B                     = 1.380649e-23; // Boltzmann constant
        constexpr double c                       = 2.99792458e8; // spped of light

        // mathematical constants
        constexpr double PI                      = 3.14159265358979323846;
    
    }

    /// @brief Generates a high-quality pseudorandom float in [0, 1) using PCG Hash.
    /// @param seed Unique global identifier (e.g., global particle index).
    /// @param salt Salt or dimension discriminator (e.g., 1 for radius, 2 for theta, etc.).
    inline float pcg_random_float(uint64_t seed, uint32_t salt)
    {
        // 1. PCG Hash core
        uint32_t combined = static_cast<uint32_t>(seed ^ (seed >> 32)) + salt * 0x9e3779b9u;
        uint32_t state = combined * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        uint32_t hash_val = (word >> 22u) ^ word;
    
        // 2. Convert to float [0, 1) with full mantissa precision
        return static_cast<float>(hash_val) / 4294967296.0f;
    }

    /// @brief Shared runtime state and resource references passed across all simulation steps.
    struct SimulationContext {
        sycl::queue &queue;                 ///< SYCL queue used for kernel submissions and memory operations
        ParticleStorage &storage;           ///< Container managing particle species and VRAM allocations
        GridField &grid;                    ///< Field solver data and spatial grid structures
        SpeciesRegistry &species_registry;  ///< Registry holding physical and numerical traits of active species
        
        double dt;                           ///< Time increment per simulation step [s]
        double current_time = 0.0f;          ///< Elapsed physical simulation time [s]
        size_t current_step = 0;            ///< Current timestep iteration index
    };
    
    /// @brief Simple 3D vector structure for spatial positions and velocity directions.
    template <typename T>
    requires std::floating_point<T>
    struct Vec3 {
        T x = 0.0f;
        T y = 0.0f;
        T z = 0.0f;

        /// @brief Constructors
        constexpr Vec3() = default;
        constexpr Vec3(T px, T py, T pz) : x(px), y(py), z(pz) {}
        template <typename U>
        constexpr Vec3(const Vec3<U>& other) 
        : x(static_cast<T>(other.x)), 
          y(static_cast<T>(other.y)), 
          z(static_cast<T>(other.z)) {}

        /// @brief Computes the dot product of two Vec3 instances.
        static inline T dot(const Vec3& a, const Vec3& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        /// @brief Statically adds two vectors together.
        static inline Vec3 add(const Vec3& a, const Vec3& b) {
            return {a.x + b.x, a.y + b.y, a.z + b.z};
        }

        /// @brief Statically subtracts two vectors together.
        static inline Vec3 sub(const Vec3& a, const Vec3& b) {
            return {a.x - b.x, a.y - b.y, a.z - b.z};
        }

        /// @brief Computes the cross product of two Vec3 instances (a x b).
        static inline Vec3 cross(const Vec3& a, const Vec3& b) {
            return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x
            };
        }
    
        /// @brief Multiplies a vector by a scalar value (v * scalar).
        static inline Vec3 scale(const Vec3& v, T scalar) {
            return {v.x * scalar, v.y * scalar, v.z * scalar};
        }
    };

    using Vector3 = Vec3<float>;

    inline Vector3 compute_effective_normal(Vector3 velocity, Vector3 raw_normal){
        Vector3 effective_normal = raw_normal;
        float dot = velocity.x * raw_normal.x + velocity.y * raw_normal.y + velocity.z * raw_normal.z;
        if(dot < 0.0f) {
            effective_normal = {-raw_normal.x, -raw_normal.y, -raw_normal.z};
        }
        return effective_normal;
    }

    /**
     * @brief Supported mesh file formats.
     */
    enum class MeshFormat {
        MSH_v4_ASCII,
        Unknown
    };

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

        inline float device_compute_element_volume(const Vector3* d_nodes, const int* element_indices) const {
            switch (val) {
                case Tetrahedron: {
                    int i0 = element_indices[0];
                    int i1 = element_indices[1];
                    int i2 = element_indices[2];
                    int i3 = element_indices[3];

                    auto r0 = static_cast<Vec3<double>>(d_nodes[i0]);
                    auto r1 = static_cast<Vec3<double>>(d_nodes[i1]);
                    auto r2 = static_cast<Vec3<double>>(d_nodes[i2]);
                    auto r3 = static_cast<Vec3<double>>(d_nodes[i3]);

                    auto v1 = Vec3<double>::sub(r1, r0);
                    auto v2 = Vec3<double>::sub(r2, r0);
                    auto v3 = Vec3<double>::sub(r3, r0);

                    auto cross_v2_v3 = Vec3<double>::cross(v2, v3);
                    double det = Vec3<double>::dot(v1, cross_v2_v3);

                    return static_cast<float>(sycl::fabs(det) / 6.0);
                }
                case Hexahedron: {
                    Vec3<double> center = {0.0, 0.0, 0.0};
                    for (int i = 0; i < 8; ++i) {
                        auto node = static_cast<Vec3<double>>(d_nodes[element_indices[i]]);
                        center.x += node.x;
                        center.y += node.y;
                        center.z += node.z;
                    }
                    center.x *= 0.125;
                    center.y *= 0.125;
                    center.z *= 0.125;

                    auto tet_vol = [&](int a, int b, int c, int d) {
                        auto r0 = static_cast<Vec3<double>>(d_nodes[element_indices[a]]);
                        auto r1 = static_cast<Vec3<double>>(d_nodes[element_indices[b]]);
                        auto r2 = static_cast<Vec3<double>>(d_nodes[element_indices[c]]);
                        auto r3 = static_cast<Vec3<double>>(d_nodes[element_indices[d]]);
                        auto v1 = Vec3<double>::sub(r1, r0);
                        auto v2 = Vec3<double>::sub(r2, r0);
                        auto v3 = Vec3<double>::sub(r3, r0);
                        double det = Vec3<double>::dot(v1, Vec3<double>::cross(v2, v3));
                        return sycl::fabs(det) / 6.0;
                    };

                    return static_cast<float>(tet_vol(0, 1, 3, 4) + 
                                              tet_vol(1, 2, 3, 6) + 
                                              tet_vol(4, 5, 6, 1) + 
                                              tet_vol(4, 6, 7, 3) + 
                                              tet_vol(1, 3, 4, 6));
                }
                case Prism: {
                    auto tet_vol = [&](int a, int b, int c, int d) {
                        auto r0 = static_cast<Vec3<double>>(d_nodes[element_indices[a]]);
                        auto r1 = static_cast<Vec3<double>>(d_nodes[element_indices[b]]);
                        auto r2 = static_cast<Vec3<double>>(d_nodes[element_indices[c]]);
                        auto r3 = static_cast<Vec3<double>>(d_nodes[element_indices[d]]);
                        auto v1 = Vec3<double>::sub(r1, r0);
                        auto v2 = Vec3<double>::sub(r2, r0);
                        auto v3 = Vec3<double>::sub(r3, r0);
                        double det = Vec3<double>::dot(v1, Vec3<double>::cross(v2, v3));
                        return sycl::fabs(det) / 6.0;
                    };

                    return static_cast<float>(tet_vol(0, 1, 2, 3) + 
                                              tet_vol(1, 4, 2, 3) + 
                                              tet_vol(2, 4, 5, 3));
                }
                case Pyramid: {
                    auto tet_vol = [&](int a, int b, int c, int d) {
                        auto r0 = static_cast<Vec3<double>>(d_nodes[element_indices[a]]);
                        auto r1 = static_cast<Vec3<double>>(d_nodes[element_indices[b]]);
                        auto r2 = static_cast<Vec3<double>>(d_nodes[element_indices[c]]);
                        auto r3 = static_cast<Vec3<double>>(d_nodes[element_indices[d]]);
                        auto v1 = Vec3<double>::sub(r1, r0);
                        auto v2 = Vec3<double>::sub(r2, r0);
                        auto v3 = Vec3<double>::sub(r3, r0);
                        double det = Vec3<double>::dot(v1, Vec3<double>::cross(v2, v3));
                        return sycl::fabs(det) / 6.0;
                    };

                    return static_cast<float>(tet_vol(0, 1, 3, 4) + 
                                              tet_vol(1, 2, 3, 4));
                }
                case BoundaryTriangle: {
                    int i0 = element_indices[0];
                    int i1 = element_indices[1];
                    int i2 = element_indices[2];

                    auto r0 = static_cast<Vec3<double>>(d_nodes[i0]);
                    auto r1 = static_cast<Vec3<double>>(d_nodes[i1]);
                    auto r2 = static_cast<Vec3<double>>(d_nodes[i2]);

                    auto v1 = Vec3<double>::sub(r1, r0);
                    auto v2 = Vec3<double>::sub(r2, r0);
                    auto cross_v = Vec3<double>::cross(v1, v2);

                    double double_area_sq = cross_v.x * cross_v.x + cross_v.y * cross_v.y + cross_v.z * cross_v.z;
                    return static_cast<float>(0.5 * sycl::sqrt(double_area_sq));
                }
                case BoundaryQuad: {
                    auto tri_area = [&](int a, int b, int c) {
                        auto r0 = static_cast<Vec3<double>>(d_nodes[element_indices[a]]);
                        auto r1 = static_cast<Vec3<double>>(d_nodes[element_indices[b]]);
                        auto r2 = static_cast<Vec3<double>>(d_nodes[element_indices[c]]);
                        auto v1 = Vec3<double>::sub(r1, r0);
                        auto v2 = Vec3<double>::sub(r2, r0);
                        auto cross_v = Vec3<double>::cross(v1, v2);
                        double double_area_sq = cross_v.x * cross_v.x + cross_v.y * cross_v.y + cross_v.z * cross_v.z;
                        return 0.5 * sycl::sqrt(double_area_sq);
                    };

                    return static_cast<float>(tri_area(0, 1, 2) + tri_area(0, 2, 3));
                }
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
                auto node = static_cast<Vec3<double>>(d_nodes[element_indices[i]]);
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
    };
}

using namespace SimulationUtils;
using Vector3 = SimulationUtils::Vec3<float>;
using Vector3d = SimulationUtils::Vec3<double>;