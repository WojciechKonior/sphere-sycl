#pragma once
#include <sycl/sycl.hpp>

/// @brief Ogólna strukturalna baza danych macierzy (domyślna dla dowolnych wymiarów)
template <typename T, int Rows, int Cols>
struct MatrixStorage {
    T data[Rows][Cols] = {{0}};
};

/// @brief Specjalizacja kontenera danych DOKŁADNIE dla wymiaru 3x1 (wektora 3D).
template <typename T>
struct MatrixStorage<T, 3, 1> {
    union {
        T data[3][1];
        struct { T x; T y; T z; };
    };

    constexpr MatrixStorage() : x(0), y(0), z(0) {}
};

/// @brief General, fully dynamic template-based matrix structure supporting arbitrary dimensions.
/// @tparam T Floating-point type (e.g., float, double).
/// @tparam Rows Number of rows in the matrix.
/// @tparam Cols Number of columns in the matrix.
template <typename T, int Rows, int Cols>
    requires std::floating_point<T> && (Rows > 0) && (Cols > 0)
struct Matrix : public MatrixStorage<T, Rows, Cols>
{
    static constexpr int kRows = Rows;
    static constexpr int kCols = Cols;

    /// @brief Default constructor. Initializes all elements to zero.
    constexpr Matrix() = default;

    /// @brief Konstruktor dla wektora 3x1 przyjmujący współrzędne (x, y, z).
    template <typename = void>
    requires(Rows == 3 && Cols == 1)
    constexpr Matrix(T px, T py, T pz) {
        this->x = px;
        this->y = py;
        this->z = pz;
    }

    /// @brief Constructs a matrix from a raw 2D array.
    constexpr Matrix(const T (&init_data)[Rows][Cols]) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                this->data[i][j] = init_data[i][j];
            }
        }
    }

    /// @brief Conversion constructor from another matrix type with potentially different floating-point precision.
    template <typename U>
    constexpr Matrix(const Matrix<U, Rows, Cols> &other) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                this->data[i][j] = static_cast<T>(other.data[i][j]);
            }
        }
    }

    /// @brief Zwraca referencję do elementu w podanym wierszu i kolumnie (wersja modyfikowalna).
    constexpr T& at(int r, int c) {
        return this->data[r][c];
    }

    /// @brief Zwraca stałą referencję do elementu w podanym wierszu i kolumnie (wersja tylko do odczytu).
    constexpr const T& at(int r, int c) const {
        return this->data[r][c];
    }

    /// @brief Computes the determinant of a square matrix of any arbitrary order using Gaussian elimination with partial pivoting.
    /// @return The calculated determinant value.
    inline T determinant() const
        requires(Rows == Cols)
    {
        Matrix temp = *this;
        T det = T(1);
        const T eps = static_cast<T>(1e-12);

        for (int i = 0; i < Rows; ++i) {
            // Szukanie elementu głównego (pivoting) dla stabilności numerycznej
            int pivot = i;
            for (int j = i + 1; j < Rows; ++j) {
                if (sycl::fabs(temp.data[j][i]) > sycl::fabs(temp.data[pivot][i])) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                for (int k = 0; k < Rows; ++k) {
                    T t = temp.data[i][k];
                    temp.data[i][k] = temp.data[pivot][k];
                    temp.data[pivot][k] = t;
                }
                det = -det;
            }

            if (sycl::fabs(temp.data[i][i]) < eps) {
                return T(0); // Macierz osobliwa
            }

            det *= temp.data[i][i];

            for (int j = i + 1; j < Rows; ++j) {
                T factor = temp.data[j][i] / temp.data[i][i];
                for (int k = i; k < Rows; ++k) {
                    temp.data[j][k] -= factor * temp.data[i][k];
                }
            }
        }
        return det;
    }

    /// @brief Computes and returns the inverse of a square matrix of any arbitrary order using Gauss-Jordan elimination.
    /// @return Inverted matrix.
    inline Matrix inverse() const
        requires(Rows == Cols)
    {
        // Macierz rozszerzona musi mieć 2 razy więcej kolumn: [A | I] -> wymiar [Rows x (2 * Rows)]
        Matrix<T, Rows, 2 * Rows> augmented;
        Matrix result;
        const T eps = static_cast<T>(1e-12);

        // Inicjalizacja macierzi rozszerzonej [A | I]
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                augmented.data[i][j] = this->data[i][j];
            }
            augmented.data[i][Rows + i] = T(1);
        }

        // Eliminacja Gaussa-Jordana
        for (int i = 0; i < Rows; ++i) {
            int pivot = i;
            for (int j = i + 1; j < Rows; ++j) {
                if (sycl::fabs(augmented.data[j][i]) > sycl::fabs(augmented.data[pivot][i])) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                for (int k = 0; k < 2 * Rows; ++k) {
                    T t = augmented.data[i][k];
                    augmented.data[i][k] = augmented.data[pivot][k];
                    augmented.data[pivot][k] = t;
                }
            }

            T pivot_val = augmented.data[i][i];
            if (sycl::fabs(pivot_val) < eps) {
                return result; // Zwraca zerową w przypadku macierzy osobliwej
            }

            // Normalizacja wiersza głównego
            for (int k = 0; k < 2 * Rows; ++k) {
                augmented.data[i][k] /= pivot_val;
            }

            // Zerowanie pozostałych elementów w kolumnie
            for (int j = 0; j < Rows; ++j) {
                if (j != i) {
                    T factor = augmented.data[j][i];
                    for (int k = 0; k < 2 * Rows; ++k) {
                        augmented.data[j][k] -= factor * augmented.data[i][k];
                    }
                }
            }
        }

        // Wyciągnięcie prawej strony [A^-1] z macierzi rozszerzonej
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result.data[i][j] = augmented.data[i][Rows + j];
            }
        }

        return result;
    }

    /// @brief Transposes the matrix (supports rectangular matrices of any dimensions).
    /// @return Transposed matrix of dimensions [Cols][Rows].
    constexpr Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result.data[j][i] = this->data[i][j];
            }
        }
        return result;
    }

    /// @brief Statically adds two matrices together element-wise (A + B).
    /// @param a First matrix.
    /// @param b Second matrix.
    /// @return Resulting matrix of the same dimensions.
    static inline Matrix add(const Matrix &a, const Matrix &b) {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }

    /// @brief Statically subtracts two matrices element-wise (A - B).
    /// @param a First matrix.
    /// @param b Second matrix.
    /// @return Resulting matrix of the same dimensions.
    static inline Matrix sub(const Matrix &a, const Matrix &b) {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    /// @brief Computes the cross product of two 3D vectors represented as 3x1 matrices (a x b).
    /// @param a First 3x1 vector matrix.
    /// @param b Second 3x1 vector matrix.
    /// @return Resulting 3x1 vector matrix representing the cross product.
    static inline Matrix cross(const Matrix &a, const Matrix &b)
        requires((Rows == 3 && Cols == 1) || (Rows == 1 && Cols == 3)) {
        Matrix result;
        if constexpr (Rows == 3 && Cols == 1) {
            // Wektor kolumnowy 3x1
            result.data[0][0] = a.data[1][0] * b.data[2][0] - a.data[2][0] * b.data[1][0];
            result.data[1][0] = a.data[2][0] * b.data[0][0] - a.data[0][0] * b.data[2][0];
            result.data[2][0] = a.data[0][0] * b.data[1][0] - a.data[1][0] * b.data[0][0];
        } else {
            // Wektor wierszowy 1x3
            result.data[0][0] = a.data[0][1] * b.data[0][2] - a.data[0][2] * b.data[0][1];
            result.data[0][1] = a.data[0][2] * b.data[0][0] - a.data[0][0] * b.data[0][2];
            result.data[0][2] = a.data[0][0] * b.data[0][1] - a.data[0][1] * b.data[0][0];
        }
        return result;
    }

    /// @brief Multiplies two matrices statically (A * B) for any valid dimensions.
    /// @tparam OtherCols Number of columns in the second matrix.
    /// @param a First matrix [Rows x SharedCols].
    /// @param b Second matrix [SharedCols x OtherCols].
    /// @return Resulting matrix [Rows x OtherCols].
    template <int SharedCols, int OtherCols>
    static inline Matrix<T, Rows, OtherCols> multiply(const Matrix<T, Rows, SharedCols> &a, const Matrix<T, SharedCols, OtherCols> &b) {
        Matrix<T, Rows, OtherCols> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < OtherCols; ++j) {
                T sum = T(0);
                for (int k = 0; k < SharedCols; ++k) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    /// @brief Computes the dot product (sum of element-wise products / Frobenius inner product) 
    /// of two matrices of identical dimensions. Returns a scalar T.
    static inline T dot(const Matrix<T, Rows, Cols>& a, const Matrix<T, Rows, Cols>& b) {
        T sum = T(0);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                sum += a.data[i][j] * b.data[i][j];
            }
        }
        return sum;
    }

    /// @brief Multiplies a matrix by a scalar value (M * scalar).
    /// @param m Input matrix.
    /// @param scalar Scalar value.
    /// @return Scaled matrix.
    static inline Matrix scale(const Matrix &m, T scalar) {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result.data[i][j] = m.data[i][j] * scalar;
            }
        }
        return result;
    }

    /// @brief Horizontally merges multiple matrices/vectors (column-wise concatenation).
    ///        All input matrices must have the exact same number of rows.
    /// @tparam MappedMatrices Parameter pack of other Matrix types.
    /// @param other Additional matrices to append on the right.
    /// @return A new Matrix with combined columns [Rows x (Cols + sum(Cols_i))].
    template <typename... MappedMatrices>
    static inline auto hmerge(const Matrix& first, const MappedMatrices&... others) {
        constexpr int TotalCols = kCols + (MappedMatrices::kCols + ...);
        
        // Wymóg: wszystkie macierze muszą mieć tę samą liczbę wierszy (Rows)
        static_assert(((MappedMatrices::kRows == kRows) && ...), 
                      "Error: All matrices passed to hmerge must have the same number of rows!");

        Matrix<T, Rows, TotalCols> result{};

        // Lambda pomocnicza do skopiowania danych macierzy do wyniku na odpowiednią pozycję kolumnową
        auto copy_to_result = [&](const auto& mat, int& start_col) {
            constexpr int MatCols = std::decay_t<decltype(mat)>::kCols;
            constexpr int MatRows = std::decay_t<decltype(mat)>::kRows;
            for (int i = 0; i < MatRows; ++i) {
                for (int j = 0; j < MatCols; ++j) {
                    result.data[i][start_col + j] = mat.data[i][j];
                }
            }
            start_col += MatCols;
        };

        int current_col = 0;
        copy_to_result(first, current_col);
        (copy_to_result(others, current_col), ...);

        return result;
    }
};