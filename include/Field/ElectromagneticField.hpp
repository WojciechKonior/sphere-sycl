#pragma once

struct ElectromagneticField {
    // Tutaj w przyszłości znajdą się sycl::accessor do pamięci pól na siatce

    /**
     * @brief Funkcja wywoływana bezpośrednio w kernelu GPU do samplowania pól w punkcie (x, y, z).
     */
    inline void device_sample(float x, float y, float z, float &Ex, float &Ey, float &Ez, float &Bx, float &By, float &Bz) const {
        // Docelowo: interpolacja z węzłów siatki FDTD (np. Yee lattice)
        Ex = 0.0f; Ey = 0.0f; Ez = 0.0f;
        Bx = 0.0f; By = 0.0f; Bz = 1.0f; // Przykładowe stałe pole magnetyczne w osi Z
    }
};