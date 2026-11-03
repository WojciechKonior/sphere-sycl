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
    };

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
}

using namespace SimulationUtils;
using Vector3 = SimulationUtils::Vec3<float>;
using Vector3d = SimulationUtils::Vec3<double>;