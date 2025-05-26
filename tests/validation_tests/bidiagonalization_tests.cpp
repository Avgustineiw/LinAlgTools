#include "../../src/algorithms/bidiagonalization.h"
#include "../../src/core/math_utils.h"
#include "../../src/core/matrix_traits.h"
#include "../../src/types/matrix.h"

#include <complex>
#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace LinAlgTools::Core;
using namespace LinAlgTools::Algorithm;

template<MatrixType M, MatrixType U, MatrixType B, MatrixType VT>
bool CheckBidiagonalization(const M& matrix, const U& unitary_u, const B& bidiagonal, const VT& unitary_vt) {
        return AreEqualMatrices(matrix, unitary_u * bidiagonal * unitary_vt) &&
               IsUnitary(unitary_u) &&
               IsUnitary(unitary_vt) &&
               IsBidiagonal(bidiagonal);
}

TEST(TEST_BIDIAGONALIZATION, SquareRealMatrix) {
        Matrix<double> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12},
                            {13, 14, 15, 16}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, RectangularMatrix) {
        Matrix<double> A = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9},
                            {10, 11, 12}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));

        Matrix<double> B_mat = {{1, 2, 3, 4},
                                {5, 6, 7, 8}};
        auto [U2, B2, VT2] = Bidiagonalization(B_mat);
        EXPECT_TRUE(CheckBidiagonalization(B_mat, U2, B2, VT2));
}

TEST(TEST_BIDIAGONALIZATION, ComplexMatrix) {
        Matrix<std::complex<double>> A = {{{1, 0}, {2, 1}, {3, 0}},
                                          {{4, -1}, {5, 0}, {6, 1}},
                                          {{7, 0}, {8, -1}, {9, 0}}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, AlreadyBidiagonal) {
        Matrix<double> A = {{1, 2, 0, 0},
                            {0, 3, 4, 0},
                            {0, 0, 5, 6}};
        auto [U, B, VT] = Bidiagonalization(A);

        Matrix<double> expected_identity_u = Matrix<double>::Identity(A.Rows());
        Matrix<double> expected_identity_vt = Matrix<double>::Identity(A.Columns());
        EXPECT_TRUE(U == expected_identity_u);
        EXPECT_TRUE(VT == expected_identity_vt);
        EXPECT_TRUE(B == A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, DiagonalMatrix) {
        Matrix<double> A = {{1, 0, 0},
                            {0, 2, 0},
                            {0, 0, 3}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, ZeroMatrix) {
        Matrix<double> A = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};
        auto [U, B, VT] = Bidiagonalization(A);

        Matrix<double> expected_identity_u = Matrix<double>::Identity(A.Rows());
        Matrix<double> expected_identity_vt = Matrix<double>::Identity(A.Columns());
        EXPECT_TRUE(U == expected_identity_u);
        EXPECT_TRUE(VT == expected_identity_vt);
        EXPECT_TRUE(B == A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, IdentityMatrix) {
        Matrix<double> A = Matrix<double>::Identity(4);
        auto [U, B, VT] = Bidiagonalization(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(4);
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(VT == expected_identity);
        EXPECT_TRUE(B == expected_identity);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, IllConditionedMatrix) {
        Matrix<double> A = {{1, 1e10, 0},
                            {1e-10, 1, 1e10},
                            {0, 1e-10, 1}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, SingleElementMatrix) {
        Matrix<double> A = {{5}};
        auto [U, B, VT] = Bidiagonalization(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(1);
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(VT == expected_identity);
        EXPECT_TRUE(B == A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, RowVector) {
        Matrix<double> A = {{1, 2, 3, 4}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, ColumnVector) {
        Matrix<double> A = {{1},
                            {2},
                            {3}};
        auto [U, B, VT] = Bidiagonalization(A);
        EXPECT_TRUE(CheckBidiagonalization(A, U, B, VT));
}

TEST(TEST_BIDIAGONALIZATION, PreserveBidiagonalStructure) {
        Matrix<double> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12}};
        auto [U, B, VT] = Bidiagonalization(A);

        for (Index i = 0; i < B.Rows(); i++) {
                for (Index j = 0; j < B.Columns(); j++) {
                        if (j != i && j != i + 1) {
                                EXPECT_NEAR(B(i, j), 0.0, 1e-10);
                        }
                }
        }
}

