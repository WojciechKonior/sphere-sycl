#pragma once

struct ElectromagneticField {
    // Tutaj w przyszłości znajdą się sycl::accessor do pamięci pól na siatce

    /**
     * @brief Funkcja wywoływana bezpośrednio w kernelu GPU do samplowania pól w punkcie (x, y, z).
     */
    inline void device_sample(Vector3 pos, Vector3& E, Vector3& B) const {
        // Docelowo: interpolacja z węzłów siatki FDTD (np. Yee lattice)
        E.x = 0.0f; E.y = 0.0f; E.z = 0.0f;
        B.x = 0.0f; B.y = 0.0f; B.z = 1.0f; // Przykładowe stałe pole magnetyczne w osi Z
    }
};