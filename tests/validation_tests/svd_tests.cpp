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

namespace {
template<MatrixType M>
bool checkSingularValues(const M& sigma) {
        using T = typename M::ElementType;
        const Index size = std::min(sigma.Rows(), sigma.Columns());
        for (Index i = 0; i < size; i++) {
                if (std::imag(sigma(i, i)) != T{0} || std::real(sigma(i, i)) < 0.0) {
                        return false;
                }
                if (i < size - 1 &&
                    std::real(sigma(i, i)) < std::real(sigma(i + 1, i + 1))) {
                        return false;
                }
        }
        return true;
}
}

template<MatrixType A, MatrixType L, MatrixType E, MatrixType R>
bool CheckSVD(const A& matrix, const L& U, const E& S, const R& VT) {
        return AreEqualMatrices(matrix, U * S * VT) &&
               IsUnitary(U) &&
               IsUnitary(VT) &&
               IsDiagonal(S) &&
               checkSingularValues(S);
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

        Matrix<double> B = {{1, 2, 3},
                            {4, 5, 6}};
        auto [U2, S2, V2] = Algorithm::NaiveSVD(B);
        EXPECT_TRUE(CheckSVD(B, U2, S2, V2));
}

TEST(TEST_SVD_DECOMPOSITION, ComplexMatrix) {
        Matrix<std::complex<double>> A = {{{1, 1}, {0, 1}},
                                          {{0, -1}, {1, 0}},
                                          {{1, 0}, {0, -1}}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, SmallValues) {
        Matrix<double> A = {{1e-10, 2e-10},
                            {3e-10, 4e-10}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, ComplexSmallValues) {
        Matrix<std::complex<double>> A = {{{1e-10, 1e-10}, {2e-10, -1e-10}},
                                          {{3e-10, -2e-10}, {4e-10, 3e-10}}};
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

TEST(TEST_SVD_DECOMPOSITION, NearlySingularMatrix) {
        Matrix<double> A = {{1, 1},
                            {1, 1 + 1e-12}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
        EXPECT_GT(S(0, 0), S(1, 1));
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

TEST(TEST_SVD_DECOMPOSITION, DiagonalMatrix) {
        Matrix<double> A = {{2, 0, 0},
                            {0, 3, 0},
                            {0, 0, 4}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));

        EXPECT_DOUBLE_EQ(S(0, 0), 4.0);
        EXPECT_DOUBLE_EQ(S(1, 1), 3.0);
        EXPECT_DOUBLE_EQ(S(2, 2), 2.0);
}

TEST(TEST_SVD_DECOMPOSITION, IllConditionedMatrix) {
        Matrix<double> A = {{1, 1e10},
                            {0, 1}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));

        EXPECT_GT(S(0, 0), S(1, 1));
}

TEST(TEST_SVD_DECOMPOSITION, ColumnVector) {
        Matrix<double> A = {{1},
                            {2},
                            {3}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, RowVector) {
        Matrix<double> A = {{1, 2, 3}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, SingleElementMatrix) {
        Matrix<double> A = {{5}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);
        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

TEST(TEST_SVD_DECOMPOSITION, PresortedSingularValues) {
        Matrix<double> A = {{3, 0},
                            {0, 2},
                            {0, 0}};
        auto [U, S, V] = Algorithm::NaiveSVD(A);

        EXPECT_TRUE(CheckSVD(A, U, S, V));
}

