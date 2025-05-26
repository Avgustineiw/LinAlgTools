#include "../../src/algorithms/hessenberg_form.h"
#include "../../src/core/math_utils.h"
#include "../../src/core/matrix_traits.h"
#include "../../src/types/matrix.h"

#include <complex>
#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace LinAlgTools::Core;
using namespace LinAlgTools::Algorithm;

template<MatrixType M, MatrixType U, MatrixType H>
bool CheckHessenberg(const M& matrix, const U& unitary, const H& hessenberg) {
        return AreEqualMatrices(matrix, unitary * hessenberg * unitary.ConjugateTransposed()) &&
               IsUnitary(unitary) &&
               IsUpperHessenberg(hessenberg);
}

TEST(TEST_HESSENBERG_FORM, SquareRealMatrix) {
        Matrix<double> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12},
                            {13, 14, 15, 16}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, ComplexMatrix) {
        Matrix<std::complex<double>> A = {{{1, 0}, {2, 1}, {3, 0}},
                                          {{4, -1}, {5, 0}, {6, 1}},
                                          {{7, 0}, {8, -1}, {9, 0}}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, SmallValues) {
        Matrix<double> A = {{1e-10, 2e-10, 3e-10},
                            {4e-10, 5e-10, 6e-10},
                            {7e-10, 8e-10, 9e-10}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, AlreadyHessenberg) {
        Matrix<double> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {0, 9, 10, 11},
                            {0, 0, 12, 13}};
        auto [U, H] = HessenbergForm(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(H == A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, DiagonalMatrix) {
        Matrix<double> A = {{1, 0, 0},
                            {0, 2, 0},
                            {0, 0, 3}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, TriangularMatrix) {
        Matrix<double> A = {{1, 2, 3},
                            {0, 4, 5},
                            {0, 0, 6}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, ZeroMatrix) {
        Matrix<double> A = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};
        auto [U, H] = HessenbergForm(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(H == A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, IdentityMatrix) {
        Matrix<double> A = Matrix<double>::Identity(4);
        auto [U, H] = HessenbergForm(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(H == A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, IllConditionedMatrix) {
        Matrix<double> A = {{1, 1e10, 0},
                            {1e-10, 1, 1e10},
                            {0, 1e-10, 1}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, SymmetricMatrix) {
        Matrix<double> A = {{4, 1, 1},
                            {1, 4, 1},
                            {1, 1, 4}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));

        EXPECT_TRUE(IsTridiagonal(H));
}

TEST(TEST_HESSENBERG_FORM, SingleElementMatrix) {
        Matrix<double> A = {{5}};
        auto [U, H] = HessenbergForm(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(1);
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(H == A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, TwoByTwoMatrix) {
        Matrix<double> A = {{1, 2},
                            {3, 4}};
        auto [U, H] = HessenbergForm(A);
        EXPECT_TRUE(CheckHessenberg(A, U, H));
}

TEST(TEST_HESSENBERG_FORM, PreserveStructure) {
        Matrix<double> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12},
                            {13, 14, 15, 16}};
        auto [U, H] = HessenbergForm(A);

        for (Index i = 2; i < H.Rows(); i++) {
                for (Index j = 0; j < i - 1; j++) {
                        EXPECT_NEAR(H(i, j), 0.0, 1e-10);
                }
        }
}

