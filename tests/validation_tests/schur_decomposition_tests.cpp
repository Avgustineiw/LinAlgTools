#include "../../src/algorithms/schur_decomposition.h"
#include "../../src/core/math_utils.h"
#include "../../src/core/matrix_traits.h"
#include "../../src/types/matrix.h"

#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace LinAlgTools::Core;
using namespace LinAlgTools::Algorithm;

namespace {
template<MatrixType M, MatrixType U, MatrixType S>
bool CheckSchur(const M& matrix, const U& unitary, const S& schur) {
        return AreEqualMatrices(matrix, unitary * schur * ConjugateTransposed(unitary)) &&
               IsUnitary(unitary) &&
               IsUpperTriangular(schur);
}
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurSquareMatrix) {
        Matrix<long double> A = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9}};
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurRealEigenvalues) {
        Matrix<long double> A = {{1, 2, 3},
                                 {0, 4, 5},
                                 {0, 0, 6}};
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurComplexEigenvalues) {
        Matrix<long double> A = {{1, -1, 0},
                                 {1, 1, 0},
                                 {0, 0, 2}};
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(AreEqualMatrices(S, A));
        EXPECT_TRUE(IsOrthogonal(U));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurSymmetricMatrix) {
        Matrix<long double> A = {{4, 1, 1},
                                 {1, 4, 1},
                                 {1, 1, 4}};
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurDiagonalMatrix) {
        Matrix<long double> A = {{1, 0, 0},
                                 {0, 2, 0},
                                 {0, 0, 3}};
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurAlreadyTriangular) {
        Matrix<long double> A = {{1, 2, 3},
                                 {0, 4, 5},
                                 {0, 0, 6}};
        auto [U, S] = RealSchur(A);

        Matrix<long double> expected_identity = Matrix<long double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(S == A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurZeroMatrix) {
        Matrix<long double> A = {{0, 0, 0},
                                 {0, 0, 0},
                                 {0, 0, 0}};
        auto [U, S] = RealSchur(A);

        Matrix<long double> expected_identity = Matrix<long double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(S == A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurIdentityMatrix) {
        Matrix<long double> A = Matrix<long double>::Identity(3);
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurNearlySingular) {
        Matrix<long double> A = {{1, 1},
                                 {1, 1 + 1e-12}};
        auto [U, S] = RealSchur(A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

