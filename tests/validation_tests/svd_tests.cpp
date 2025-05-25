#include "../../src/algorithms/svd.h"
#include "../../src/core/math_utils.h"
#include "../../src/core/matrix_traits.h"
#include "../../src/types/matrix.h"

#include <algorithm>
#include <complex>
#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace LinAlgTools::Core;
using namespace LinAlgTools::Algorithm;

template<MatrixType M>
bool checkSingularValues(const M& sigma) {
        using T = typename M::ElementType;
        int32_t size = std::min(sigma.Rows(), sigma.Columns());
        for (int32_t i = 0; i < size; i++) {
                if (std::imag(sigma(i, i) != T{0}) ||
                    std::real(sigma(i, i)) < 0.0) {
                        return false;
                }
                if (i < size - 1 &&
                    std::real(sigma(i, i)) < std::real(sigma(i + 1, i + 1))) {
                        return false;
                }
        }

        return true;
}

template<MatrixType M, MatrixType U, MatrixType S, MatrixType V>
bool CheckSVD(const M& matrix, const U& unitary_u, const S& sigma, const V& unitary_v) {
        return AreEqualMatrices(matrix, unitary_u * sigma * unitary_v) &&
               IsUnitary(unitary_u) &&
               IsUnitary(unitary_v) &&
               IsDiagonal(sigma) &&
               checkSingularValues(sigma);
}

TEST(TEST_SVD_DECOMPOSITION, SquareRealMatrix) {
        Matrix<double> A = {{1, 2, 3},
                            {2, -4, -6},
                            {3, -6, 9}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, RectangularRealMatrix) {
        Matrix<double> A = {{1, 2},
                            {3, 4},
                            {5, 6}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, ComplexMatrix) {
        Matrix<std::complex<double>> A = {{{1, 0}, {0, 1}},
                                          {{0, -1}, {1, 0}}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, SingularMatrix) {
        Matrix<double> A = {{1, 1},
                            {1, 1}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
        EXPECT_NEAR(S(1, 1), 0.0, 1e-10);
}

TEST(TEST_SVD_DECOMPOSITION, IdentityMatrix) {
        Matrix<double> A = Matrix<double>::Identity(3);
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
        EXPECT_TRUE(U == A && S == A && V == A);
}

TEST(TEST_SVD_DECOMPOSITION, ZeroMatrix) {
        Matrix<double> A = {{0, 0},
                            {0, 0}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
        EXPECT_TRUE(S == A);
}

TEST(TEST_SVD_DECOMPOSITION, IllConditionedMatrix) {
        Matrix<double> A = {{1, 1e10},
                            {0, 1}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

