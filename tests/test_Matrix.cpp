#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "Matrix.hpp" // Załóżmy, że pod takim nagłówkiem trzymasz nową strukturę Matrix

/**
 * @class MatrixTest
 * @brief Test fixture for Matrix class unit tests, maintaining a SYCL queue for potential future GPU tests.
 */
class MatrixTest : public ::testing::Test {
protected:
    /// SYCL queue used for potential GPU memory allocation and execution within tests.
    sycl::queue queue;

    /**
     * @brief Sets up the test environment by creating a default SYCL queue.
     */
    void SetUp() override {
        try {
            queue = sycl::queue(sycl::gpu_selector_v);
        } catch (const sycl::exception &e) {
            // Jeśli urządzenie GPU nie jest dostępne w środowisku testowym CPU/lokalnym,
            // możemy fallbackować na domyślny selektor (np. host_selector_v lub default_selector_v),
            // żeby testy się nie wywalały, jeśli nie masz podpiętej karty zgodnej z SYCL.
            try {
                queue = sycl::queue(sycl::default_selector_v);
            } catch (...) {
                FAIL() << "Failed to create SYCL queue: " << e.what();
            }
        }
    }

    /**
     * @brief Cleans up the test environment after each test execution.
     */
    void TearDown() override {
        try {
            queue.wait();
        } catch (const sycl::exception &e) {
            ADD_FAILURE() << "SYCL exception during TearDown wait: " << e.what();
        }
    }
};

/**
 * @test Verifies all 3D vector operations (add, sub, scale, cross, dot product via multiplication)
 * using Matrix<float, 3, 1> across multiple sample vectors against analytical expectations.
 */
TEST_F(MatrixTest, Vector3DOperationsVerification) {
    // Definicja przykładowych wektorów testowych (macierze 3x1)
    using Vec3M = Matrix<float, 3, 1>;

    Vec3M v1({ {1.0f}, {2.0f}, {3.0f} });
    Vec3M v2({ {4.0f}, {5.0f}, {6.0f} });
    Vec3M v3({ {1.0f}, {0.0f}, {0.0f} });
    Vec3M v4({ {0.0f}, {1.0f}, {0.0f} });
    Vec3M v5({ {-2.0f}, {3.0f}, {1.0f} });
    Vec3M v6({ {1.0f}, {-1.0f}, {2.0f} });

    // --- 1. Test dodawania (v1 + v2) ---
    // Teoria: [1+4, 2+5, 3+6] = [5, 7, 9]
    Vec3M add_res = Vec3M::add(v1, v2);
    EXPECT_NEAR(add_res.data[0][0], 5.0f, 1e-5f);
    EXPECT_NEAR(add_res.data[1][0], 7.0f, 1e-5f);
    EXPECT_NEAR(add_res.data[2][0], 9.0f, 1e-5f);

    // --- 2. Test odejmowania (v2 - v1) ---
    // Teoria: [4-1, 5-2, 6-3] = [3, 3, 3]
    Vec3M sub_res = Vec3M::sub(v2, v1);
    EXPECT_NEAR(sub_res.data[0][0], 3.0f, 1e-5f);
    EXPECT_NEAR(sub_res.data[1][0], 3.0f, 1e-5f);
    EXPECT_NEAR(sub_res.data[2][0], 3.0f, 1e-5f);

    // --- 3. Test skalowania (v1 * 2.5) ---
    // Teoria: [1*2.5, 2*2.5, 3*2.5] = [2.5, 5.0, 7.5]
    Vec3M scale_res = Vec3M::scale(v1, 2.5f);
    EXPECT_NEAR(scale_res.data[0][0], 2.5f, 1e-5f);
    EXPECT_NEAR(scale_res.data[1][0], 5.0f, 1e-5f);
    EXPECT_NEAR(scale_res.data[2][0], 7.5f, 1e-5f);

    // --- 4. Test iloczynu wektorowego (v3 x v4 -> osie X x Y = Z) ---
    // Teoria: [1,0,0] x [0,1,0] = [0, 0, 1]
    Vec3M cross_res1 = Vec3M::cross(v3, v4);
    EXPECT_NEAR(cross_res1.data[0][0], 0.0f, 1e-5f);
    EXPECT_NEAR(cross_res1.data[1][0], 0.0f, 1e-5f);
    EXPECT_NEAR(cross_res1.data[2][0], 1.0f, 1e-5f);

    // Drugi przypadek iloczynu wektorowego (v5 x v6)
    // v5 = [-2, 3, 1], v6 = [1, -1, 2]
    // cx = (3*2) - (1*-1) = 6 + 1 = 7
    // cy = (1*1) - (-2*2) = 1 + 4 = 5
    // cz = (-2*-1) - (3*1) = 2 - 3 = -1
    Vec3M cross_res2 = Vec3M::cross(v5, v6);
    EXPECT_NEAR(cross_res2.data[0][0],  7.0f, 1e-5f);
    EXPECT_NEAR(cross_res2.data[1][0],  5.0f, 1e-5f);
    EXPECT_NEAR(cross_res2.data[2][0], -1.0f, 1e-5f);

    // --- 5. Test iloczynu skalarnego (v1 . v2) realizowany przez transpozycję i mnożenie ---
    // v1^T (1x3) * v2 (3x1) daje macierz [1x1] o wartości równej iloczynowi skalarnemu
    // Teoria: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    auto v1_t = v1.transpose(); // Typ: Matrix<float, 1, 3>
    auto dot_matrix = Matrix<float, 1, 3>::multiply(v1_t, v2);
    float dot_val = dot_matrix.data[0][0];
    EXPECT_NEAR(dot_val, 32.0f, 1e-5f);

    // Sprawdzenie dla v5 i v6: -2*1 + 3*(-1) + 1*2 = -2 - 3 + 2 = -3
    auto v5_t = v5.transpose();
    float dot_val2 = Matrix<float, 1, 3>::multiply(v5_t, v6).data[0][0];
    EXPECT_NEAR(dot_val2, -3.0f, 1e-5f);

    std::cout << "[Test Sukces] Wszystkie operacje na wektorach 3D zgodne z teorią!\n";
}

/**
 * @test Verifies all 3D matrix (3x3) operations (add, sub, multiply, scale, transpose, determinant, inverse)
 * across multiple sample matrices against analytical expectations.
 */
TEST_F(MatrixTest, Matrix3x3OperationsVerification) {
    using Mat3x3 = Matrix<float, 3, 3>;

    // 1. Definicja przykładowych macierzy testowych
    // Macierz ogólna M1 o niezerowym i łatwym do policzenia wyznaczniku (det = 1.0)
    // M1 = [ 1, 2, 3 ]
    //      [ 0, 1, 4 ]
    //      [ 5, 6, 0 ]
    Mat3x3 m1({
        {1.0f, 2.0f, 3.0f},
        {0.0f, 1.0f, 4.0f},
        {5.0f, 6.0f, 0.0f}
    });

    // Macierz ogólna M2
    // M2 = [ 2, 0, 1 ]
    //      [ 3, 1, 0 ]
    //      [ 0, 2, 1 ]
    Mat3x3 m2({
        {2.0f, 0.0f, 1.0f},
        {3.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 1.0f}
    });

    // Macierz diagonalna D (dla łatwej weryfikacji wyznacznika i odwrotności)
    // D = [ 2, 0, 0 ]
    //     [ 0, 4, 0 ]
    //     [ 0, 0, 5 ]
    Mat3x3 d_mat({
        {2.0f, 0.0f, 0.0f},
        {0.0f, 4.0f, 0.0f},
        {0.0f, 0.0f, 5.0f}
    });

    // --- 1. Test dodawania (m1 + m2) ---
    // Teoria: [1+2, 2+0, 3+1] = [3, 2, 4]
    //         [0+3, 1+1, 4+0] = [3, 2, 4]
    //         [5+0, 6+2, 0+1] = [5, 8, 1]
    Mat3x3 add_res = Mat3x3::add(m1, m2);
    EXPECT_NEAR(add_res.data[0][0], 3.0f, 1e-5f);
    EXPECT_NEAR(add_res.data[0][1], 2.0f, 1e-5f);
    EXPECT_NEAR(add_res.data[0][2], 4.0f, 1e-5f);
    EXPECT_NEAR(add_res.data[2][2], 1.0f, 1e-5f);

    // --- 2. Test odejmowania (m1 - m2) ---
    // Teoria: [1-2, 2-0, 3-1] = [-1, 2, 2]
    //         [0-3, 1-1, 4-0] = [-3, 0, 4]
    //         [5-0, 6-2, 0-1] = [ 5, 4,-1]
    Mat3x3 sub_res = Mat3x3::sub(m1, m2);
    EXPECT_NEAR(sub_res.data[0][0], -1.0f, 1e-5f);
    EXPECT_NEAR(sub_res.data[0][1],  2.0f, 1e-5f);
    EXPECT_NEAR(sub_res.data[2][2], -1.0f, 1e-5f);

    // --- 3. Test skalowania (d_mat * 2.0) ---
    // Teoria: [4, 0, 0], [0, 8, 0], [0, 0, 10]
    Mat3x3 scale_res = Mat3x3::scale(d_mat, 2.0f);
    EXPECT_NEAR(scale_res.data[0][0], 4.0f, 1e-5f);
    EXPECT_NEAR(scale_res.data[1][1], 8.0f, 1e-5f);
    EXPECT_NEAR(scale_res.data[2][2], 10.0f, 1e-5f);

    // --- 4. Test transpozycji (m1^T) ---
    // Teoria: wiersze stają się kolumnami
    // [ 1, 0, 5 ]
    // [ 2, 1, 6 ]
    // [ 3, 4, 0 ]
    Mat3x3 trans_res = m1.transpose();
    EXPECT_NEAR(trans_res.data[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(trans_res.data[0][1], 0.0f, 1e-5f);
    EXPECT_NEAR(trans_res.data[0][2], 5.0f, 1e-5f);
    EXPECT_NEAR(trans_res.data[1][0], 2.0f, 1e-5f);
    EXPECT_NEAR(trans_res.data[2][1], 4.0f, 1e-5f);

    // --- 5. Test wyznacznika (determinant) ---
    // Dla m1: det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5) = -24 + 40 - 15 = 1.0
    EXPECT_NEAR(m1.determinant(), 1.0f, 1e-5f);

    // Dla d_mat (macierz diagonalna): det = 2 * 4 * 5 = 40.0
    EXPECT_NEAR(d_mat.determinant(), 40.0f, 1e-5f);

    // --- 6. Test mnożenia macierzy (m1 * m2) ---
    // m1 * m2:
    // row 0: [1*2 + 2*3 + 3*0,  1*0 + 2*1 + 3*2,  1*1 + 2*0 + 3*1] = [8, 8, 4]
    // row 1: [0*2 + 1*3 + 4*0,  0*0 + 1*1 + 4*2,  0*1 + 1*0 + 4*1] = [3, 9, 4]
    // row 2: [5*2 + 6*3 + 0*0,  5*0 + 6*1 + 0*2,  5*1 + 6*0 + 0*1] = [28, 6, 5]
    Mat3x3 mul_res = Mat3x3::multiply(m1, m2);
    EXPECT_NEAR(mul_res.data[0][0],  8.0f, 1e-5f);
    EXPECT_NEAR(mul_res.data[0][1],  8.0f, 1e-5f);
    EXPECT_NEAR(mul_res.data[0][2],  4.0f, 1e-5f);
    EXPECT_NEAR(mul_res.data[1][0],  3.0f, 1e-5f);
    EXPECT_NEAR(mul_res.data[1][1],  9.0f, 1e-5f);
    EXPECT_NEAR(mul_res.data[2][0], 28.0f, 1e-5f);

    // --- 7. Test odwracania macierzy (inverse) ---
    // Sprawdzamy dla macierzy diagonalnej d_mat (odwrotność to [0.5, 0, 0; 0, 0.25, 0; 0, 0.2, 0.2])
    Mat3x3 d_inv = d_mat.inverse();
    EXPECT_NEAR(d_inv.data[0][0], 0.5f,  1e-5f);
    EXPECT_NEAR(d_inv.data[1][1], 0.25f, 1e-5f);
    EXPECT_NEAR(d_inv.data[2][2], 0.2f,  1e-5f);
    EXPECT_NEAR(d_inv.data[0][1], 0.0f,  1e-5f);

    // Test uniwersalny własności odwrotności: M1 * M1^-1 powinno dać macierz jednostkową I
    Mat3x3 m1_inv = m1.inverse();
    Mat3x3 identity_check = Mat3x3::multiply(m1, m1_inv);

    // Sprawdzamy czy na przekątnej mamy 1.0, a poza nią 0.0
    EXPECT_NEAR(identity_check.data[0][0], 1.0f, 1e-4f);
    EXPECT_NEAR(identity_check.data[1][1], 1.0f, 1e-4f);
    EXPECT_NEAR(identity_check.data[2][2], 1.0f, 1e-4f);

    EXPECT_NEAR(identity_check.data[0][1], 0.0f, 1e-4f);
    EXPECT_NEAR(identity_check.data[1][0], 0.0f, 1e-4f);
    EXPECT_NEAR(identity_check.data[2][0], 0.0f, 1e-4f);

    std::cout << "[Test Sukces] Wszystkie operacje na macierzach 3x3 zgodne z teorią!\n";
}

/**
 * @test Verifies that Vector3 (alias to Matrix<float, 3, 1>) fully supports 
 * direct coordinate access (.x, .y, .z) alongside all matrix and vector operations 
 * (add, sub, scale, cross, dot, transpose, matrix-vector multiplication).
 */
TEST_F(MatrixTest, Vector3SpecializationComprehensiveVerification) {
    using Vector3 = Matrix<float, 3, 1>;

    // Vector3 to nasz alias dla Matrix<float, 3, 1> z dostępem do pól .x, .y, .z
    Vector3 v1(1.0f, 2.0f, 3.0f);
    Vector3 v2(4.0f, 5.0f, 6.0f);
    Vector3 v3(2.0f, -1.0f, 3.0f);

    // --- 1. Weryfikacja bezpośredniego dostępu do pól .x, .y, .z (zgodność z unią) ---
    EXPECT_NEAR(v1.x, 1.0f, 1e-5f);
    EXPECT_NEAR(v1.y, 2.0f, 1e-5f);
    EXPECT_NEAR(v1.z, 3.0f, 1e-5f);

    // Test modyfikacji przez pola
    Vector3 v_mod(0.0f, 0.0f, 0.0f);
    v_mod.x = 10.0f;
    v_mod.y = 20.0f;
    v_mod.z = 30.0f;
    EXPECT_NEAR(v_mod.data[0][0], 10.0f, 1e-5f);
    EXPECT_NEAR(v_mod.data[1][0], 20.0f, 1e-5f);
    EXPECT_NEAR(v_mod.data[2][0], 30.0f, 1e-5f);

    // --- 2. Test dodawania (add) ---
    // [1+4, 2+5, 3+6] = [5, 7, 9]
    Vector3 sum_vec = Vector3::add(v1, v2);
    EXPECT_NEAR(sum_vec.x, 5.0f, 1e-5f);
    EXPECT_NEAR(sum_vec.y, 7.0f, 1e-5f);
    EXPECT_NEAR(sum_vec.z, 9.0f, 1e-5f);

    // --- 3. Test odejmowania (sub) ---
    // [4-1, 5-2, 6-3] = [3, 3, 3]
    Vector3 sub_vec = Vector3::sub(v2, v1);
    EXPECT_NEAR(sub_vec.x, 3.0f, 1e-5f);
    EXPECT_NEAR(sub_vec.y, 3.0f, 1e-5f);
    EXPECT_NEAR(sub_vec.z, 3.0f, 1e-5f);

    // --- 4. Test skalowania (scale) ---
    // [1*3, 2*3, 3*3] = [3, 6, 9]
    Vector3 scaled_vec = Vector3::scale(v1, 3.0f);
    EXPECT_NEAR(scaled_vec.x, 3.0f, 1e-5f);
    EXPECT_NEAR(scaled_vec.y, 6.0f, 1e-5f);
    EXPECT_NEAR(scaled_vec.z, 9.0f, 1e-5f);

    // --- 5. Test iloczynu wektorowego (cross) ---
    // v1 = [1, 2, 3], v3 = [2, -1, 3]
    // cx = (2*3) - (3*-1) = 6 + 3 = 9
    // cy = (3*2) - (1*3)  = 6 - 3 = 3
    // cz = (1*-1) - (2*2) = -1 - 4 = -5
    Vector3 cross_vec = Vector3::cross(v1, v3);
    EXPECT_NEAR(cross_vec.x,  9.0f, 1e-5f);
    EXPECT_NEAR(cross_vec.y,  3.0f, 1e-5f);
    EXPECT_NEAR(cross_vec.z, -5.0f, 1e-5f);

    // --- 6. Test iloczynu skalarnego (dot - wersja macierzowa dla równych wymiarów) ---
    // Teoria: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    float dot_res = Vector3::dot(v1, v2);
    EXPECT_NEAR(dot_res, 32.0f, 1e-5f);

    // Drugi przykład dla v1 i v3: 1*2 + 2*(-1) + 3*3 = 2 - 2 + 9 = 9
    float dot_res2 = Vector3::dot(v1, v3);
    EXPECT_NEAR(dot_res2, 9.0f, 1e-5f);

    // --- 7. Test transpozycji (transpose) ---
    // Vector3 (3x1) po transpozycji staje się macierzą wierszową (1x3)
    auto v1_row = v1.transpose();
    static_assert(std::is_same_v<decltype(v1_row), Matrix<float, 1, 3>>, "Transpose of 3x1 must be 1x3!");
    EXPECT_NEAR(v1_row.data[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(v1_row.data[0][1], 2.0f, 1e-5f);
    EXPECT_NEAR(v1_row.data[0][2], 3.0f, 1e-5f);

    // --- 8. Test mnożenia macierzowego (multiply) ---
    // Przemnożenie macierzy 3x3 przez nasz wektor kolumnowy Vector3 (3x1) -> wynik to nowy Vector3 (3x1)
    // M = [ 1, 0, 0 ]
    //     [ 0, 2, 0 ]
    //     [ 0, 0, 3 ]
    Matrix<float, 3, 3> scaling_mat({
        {1.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 3.0f}
    });

    // M * v1 = [1*1 + 0*2 + 0*3,  0*1 + 2*2 + 0*3,  0*1 + 0*2 + 3*3] = [1, 4, 9]
    auto mat_vec_mul = Matrix<float, 3, 3>::multiply(scaling_mat, v1);
    
    // Sprawdzamy czy wynik zachowuje strukturę Vector3 / wymiar 3x1
    EXPECT_NEAR(mat_vec_mul.data[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(mat_vec_mul.data[1][0], 4.0f, 1e-5f);
    EXPECT_NEAR(mat_vec_mul.data[2][0], 9.0f, 1e-5f);

    std::cout << "[Test Sukces] Wszystkie operacje na typie Vector3 (specjalizacja 3x1) przeszły pomyślnie!\n";
}

/**
 * @test Verifies that Vector3 (alias to Matrix<float, 3, 1>) fully supports 
 * direct coordinate access (.x, .y, .z) alongside all matrix and vector operations 
 * (add, sub, scale, cross, dot, transpose, matrix-vector multiplication, hmerge).
 */
TEST_F(MatrixTest, Vector3SpecializationHmergeVerification) {
    using Vector3 = Matrix<float, 3, 1>;
    
    // Vector3 to nasz alias dla Matrix<float, 3, 1> z dostępem do pól .x, .y, .z
    Vector3 v1(1.0f, 2.0f, 3.0f);
    Vector3 v2(4.0f, 5.0f, 6.0f);
    Vector3 v3(2.0f, -1.0f, 3.0f);

    // --- 1. Weryfikacja bezpośredniego dostępu do pól .x, .y, .z (zgodność z unią) ---
    EXPECT_NEAR(v1.x, 1.0f, 1e-5f);
    EXPECT_NEAR(v1.y, 2.0f, 1e-5f);
    EXPECT_NEAR(v1.z, 3.0f, 1e-5f);

    // Test modyfikacji przez pola
    Vector3 v_mod(0.0f, 0.0f, 0.0f);
    v_mod.x = 10.0f;
    v_mod.y = 20.0f;
    v_mod.z = 30.0f;
    EXPECT_NEAR(v_mod.data[0][0], 10.0f, 1e-5f);
    EXPECT_NEAR(v_mod.data[1][0], 20.0f, 1e-5f);
    EXPECT_NEAR(v_mod.data[2][0], 30.0f, 1e-5f);

    // --- 2. Test dodawania (add) ---
    // [1+4, 2+5, 3+6] = [5, 7, 9]
    Vector3 sum_vec = Vector3::add(v1, v2);
    EXPECT_NEAR(sum_vec.x, 5.0f, 1e-5f);
    EXPECT_NEAR(sum_vec.y, 7.0f, 1e-5f);
    EXPECT_NEAR(sum_vec.z, 9.0f, 1e-5f);

    // --- 3. Test odejmowania (sub) ---
    // [4-1, 5-2, 6-3] = [3, 3, 3]
    Vector3 sub_vec = Vector3::sub(v2, v1);
    EXPECT_NEAR(sub_vec.x, 3.0f, 1e-5f);
    EXPECT_NEAR(sub_vec.y, 3.0f, 1e-5f);
    EXPECT_NEAR(sub_vec.z, 3.0f, 1e-5f);

    // --- 4. Test skalowania (scale) ---
    // [1*3, 2*3, 3*3] = [3, 6, 9]
    Vector3 scaled_vec = Vector3::scale(v1, 3.0f);
    EXPECT_NEAR(scaled_vec.x, 3.0f, 1e-5f);
    EXPECT_NEAR(scaled_vec.y, 6.0f, 1e-5f);
    EXPECT_NEAR(scaled_vec.z, 9.0f, 1e-5f);

    // --- 5. Test iloczynu wektorowego (cross) ---
    // v1 = [1, 2, 3], v3 = [2, -1, 3]
    // cx = (2*3) - (3*-1) = 6 + 3 = 9
    // cy = (3*2) - (1*3)  = 6 - 3 = 3
    // cz = (1*-1) - (2*2) = -1 - 4 = -5
    Vector3 cross_vec = Vector3::cross(v1, v3);
    EXPECT_NEAR(cross_vec.x,  9.0f, 1e-5f);
    EXPECT_NEAR(cross_vec.y,  3.0f, 1e-5f);
    EXPECT_NEAR(cross_vec.z, -5.0f, 1e-5f);

    // --- 6. Test iloczynu skalarnego (dot - wersja macierzowa dla równych wymiarów) ---
    // Teoria: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    float dot_res = Vector3::dot(v1, v2);
    EXPECT_NEAR(dot_res, 32.0f, 1e-5f);

    // Drugi przykład dla v1 i v3: 1*2 + 2*(-1) + 3*3 = 2 - 2 + 9 = 9
    float dot_res2 = Vector3::dot(v1, v3);
    EXPECT_NEAR(dot_res2, 9.0f, 1e-5f);

    // --- 7. Test transpozycji (transpose) ---
    // Vector3 (3x1) po transpozycji staje się macierzą wierszową (1x3)
    auto v1_row = v1.transpose();
    static_assert(std::is_same_v<decltype(v1_row), Matrix<float, 1, 3>>, "Transpose of 3x1 must be 1x3!");
    EXPECT_NEAR(v1_row.data[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(v1_row.data[0][1], 2.0f, 1e-5f);
    EXPECT_NEAR(v1_row.data[0][2], 3.0f, 1e-5f);

    // --- 8. Test mnożenia macierzowego (multiply) ---
    // Przemnożenie macierzy 3x3 przez nasz wektor kolumnowy Vector3 (3x1) -> wynik to nowy Vector3 (3x1)
    Matrix<float, 3, 3> scaling_mat({
        {1.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 3.0f}
    });

    // M * v1 = [1*1 + 0*2 + 0*3,  0*1 + 2*2 + 0*3,  0*1 + 0*2 + 3*3] = [1, 4, 9]
    auto mat_vec_mul = Matrix<float, 3, 3>::multiply(scaling_mat, v1);
    
    EXPECT_NEAR(mat_vec_mul.data[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(mat_vec_mul.data[1][0], 4.0f, 1e-5f);
    EXPECT_NEAR(mat_vec_mul.data[2][0], 9.0f, 1e-5f);

    // --- 9. Test łączenia wektorów kolumnowych za pomocą hmerge ---
    Vector3 col1(1.0f, 4.0f, 7.0f);
    Vector3 col2(2.0f, 5.0f, 8.0f);
    Vector3 col3(3.0f, 6.0f, 9.0f);

    auto merged_mat = Vector3::hmerge(col1, col2, col3);

    // Weryfikacja typu w czasie kompilacji
    static_assert(std::is_same_v<decltype(merged_mat), Matrix<float, 3, 3>>, "hmerge of three 3x1 vectors must result in Matrix<float, 3, 3>!");

    // Weryfikacja poprawności danych w połączonej macierzy (kolumna po kolumnie)
    EXPECT_NEAR(merged_mat.data[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(merged_mat.data[1][0], 4.0f, 1e-5f);
    EXPECT_NEAR(merged_mat.data[2][0], 7.0f, 1e-5f);

    EXPECT_NEAR(merged_mat.data[0][1], 2.0f, 1e-5f);
    EXPECT_NEAR(merged_mat.data[1][1], 5.0f, 1e-5f);
    EXPECT_NEAR(merged_mat.data[2][1], 8.0f, 1e-5f);

    EXPECT_NEAR(merged_mat.data[0][2], 3.0f, 1e-5f);
    EXPECT_NEAR(merged_mat.data[1][2], 6.0f, 1e-5f);
    EXPECT_NEAR(merged_mat.data[2][2], 9.0f, 1e-5f);

    std::cout << "[Test Sukces] Wszystkie operacje, w tym hmerge, przeszły pomyślnie!\n";
}