#pragma once

struct ElectromagneticField {
    // Tutaj w przyszłości znajdą się sycl::accessor do pamięci pól na siatce

    /**
     * @brief Funkcja wywoływana bezpośrednio w kernelu GPU do samplowania pól w punkcie (x, y, z).
     */
    template <typename Float>
    requires std::floating_point<Float>
    inline void device_sample(const Vec3<Float> &pos, Vec3<Float>& E, Vec3<Float>& B) const {
        // Docelowo: interpolacja z węzłów siatki FDTD (np. Yee lattice)
        E.x = static_cast<Float>(0.0f);
        E.y = static_cast<Float>(0.0f);
        E.z = static_cast<Float>(0.0f);
        B.x = static_cast<Float>(0.0f);
        B.y = static_cast<Float>(0.0f);
        B.z = static_cast<Float>(1.0f); // Przykładowe stałe pole magnetyczne w osi Z
    }
};