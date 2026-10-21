#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include "SimulationUtils.hpp"

/**
 * @class FlatQuadShape
 * @brief Planar quadrilateral shape defined by 4 coplanar vertices.
 */
class FlatQuadShape {
private:
    Vector3 p0_, p1_, p2_, p3_;
    Vector3 normal_;
    float thickness_; ///< Layer thickness = (velocity . normal) * dt

    /**
     * @brief Helper function to compute and validate the normal vector from 4 vertices.
     * @param p0, p1, p2, p3 Four vertices defining the quad.
     * @return Computed unit normal vector.
     * @throws std::runtime_error if points are not coplanar or form a degenerate shape.
     */
    static Vector3 calculate_normal(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3) {
        Vector3 v1 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
        Vector3 v2 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
        Vector3 v3 = {p3.x - p0.x, p3.y - p0.y, p3.z - p0.z};

        float cross_x = v1.y * v2.z - v1.z * v2.y;
        float cross_y = v1.z * v2.x - v1.x * v2.z;
        float cross_z = v1.x * v2.y - v1.y * v2.x;
        float triple_product = v3.x * cross_x + v3.y * cross_y + v3.z * cross_z;

        // Sprawdzenie współpłaszczyznowości (Scalar Triple Product)
        if (std::abs(triple_product) > 1e-3f) {
            throw std::runtime_error("FlatQuadShape: The 4 provided points are not coplanar!");
        }

        float len = std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        if (len == 0.0f) {
            throw std::runtime_error("FlatQuadShape: Degenerate quad points resulting in zero area!");
        }

        return {cross_x / len, cross_y / len, cross_z / len};
    }

    /**
     * @brief Helper to compute a single triangle area in 3D.
     * @param a, b, c Vertices of the triangle.
     * @return Surface area of the triangle [m^2].
     */
    static float compute_triangle_area(Vector3 a, Vector3 b, Vector3 c) {
        float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
        float acx = c.x - a.x, acy = c.y - a.y, acz = c.z - a.z;
        float cx = aby * acz - abz * acy;
        float cy = abz * acx - abx * acz;
        float cz = abx * acy - aby * acx;
        return 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
    }

public:
    /**
     * @brief Constructs a pure geometry flat quad from 4 vertices.
     * @param p0 First vertex.
     * @param p1 Second vertex.
     * @param p2 Third vertex.
     * @param p3 Fourth vertex.
     */
    FlatQuadShape(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, double dt)
        : p0_(p0), p1_(p1), p2_(p2), p3_(p3),
          normal_(calculate_normal(p0, p1, p2, p3)) {}

    /**
     * @brief Aligns the quad's orientation and winding order to match the flow velocity vector.
     * 
     * If the normal points against the flow velocity vector (dot product < 0), 
     * vertices p1 and p3 are swapped and the normal is inverted.
     * 
     * @param velocity 3D flow velocity vector [m/s].
     */
    void orient_towards_velocity(const Vector3 &velocity) {
        float dot_product = velocity.x * normal_.x + velocity.y * normal_.y + velocity.z * normal_.z;
        if (dot_product < 0.0f) {
            std::swap(p1_, p3_);
            normal_ = calculate_normal(p0_, p1_, p2_, p3_);
        }
    }

    /**
     * @brief Computes the total surface area dynamically by decomposing into two triangles.
     * @return Total surface area [m^2].
     */
    double compute_area() const {
        float area_1 = compute_triangle_area(p0_, p1_, p2_);
        float area_2 = compute_triangle_area(p0_, p2_, p3_);
        return static_cast<double>(area_1 + area_2);
    }

    /**
     * @brief Separate getter for area.
     * @return Total surface area [m^2].
     */
    double get_area() const {
        return compute_area();
    }

    /**
     * @brief Generates a random position on the device within the planar quad layer.
     * 
     * @param seed_idx Unique seed index per particle.
     * @param thickness Injection layer thickness [m].
     * @param normal Inward normal vector defining the injection depth direction.
     * @return Generated 3D Cartesian position vector.
     */
    inline Vector3 device_generate_position(uint64_t seed_idx, float thickness, Vector3 normal) const {
        float xi_tri = SimulationUtils::pcg_random_float(seed_idx, 1);
        float xi1    = SimulationUtils::pcg_random_float(seed_idx, 2);
        float xi2    = SimulationUtils::pcg_random_float(seed_idx, 3);
        float xi_h   = SimulationUtils::pcg_random_float(seed_idx, 4);

        float area_1 = compute_triangle_area(p0_, p1_, p2_);
        float area_2 = compute_triangle_area(p0_, p2_, p3_);
        double total = static_cast<double>(area_1 + area_2);
        float p1_weight = area_1 / static_cast<float>(total);

        Vector3 a, b, c;
        if (xi_tri < p1_weight) {
            a = p0_; b = p1_; c = p2_;
        } else {
            a = p0_; b = p2_; c = p3_;
        }

        // Współrzędne barycentryczne wewnątrz trójkąta
        float su1 = sycl::sqrt(xi1);
        float u = 1.0f - su1;
        float v = su1 * (1.0f - xi2);
        float w = su1 * xi2;

        float base_px = u * a.x + v * b.x + w * c.x;
        float base_py = u * a.y + v * b.y + w * c.y;
        float base_pz = u * a.z + v * b.z + w * c.z;

        // Głębokość w warstwie [0, thickness_] wzdłuż wektora normalnego
        float h = xi_h * thickness;

        return {
            base_px + h * normal.x,
            base_py + h * normal.y,
            base_pz + h * normal.z
        };
    }

    /**
     * @brief Returns the inward normal vector.
     * @return Current unit normal vector.
     */
    inline Vector3 device_get_inward_normal() const {
        return normal_;
    }
};