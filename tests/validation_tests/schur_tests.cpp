#include "../../src/algorithms/schur_decomposition.h"
#include "../../src/core/math_utils.h"
#include "../../src/core/matrix_traits.h"
#include "../../src/types/matrix.h"

#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace LinAlgTools::Core;
using namespace LinAlgTools::Algorithm;

template<typename T = double>
using Index = Core::Indices::Index;

template<MatrixType M, MatrixType U, MatrixType S>
bool CheckSchur(const M& matrix, const U& unitary, const S& schur) {
        Matrix<typename M::ElementType> reconstruction = unitary * schur * unitary.ConjugateTransposed();

        Matrix<typename M::ElementType> identity = Matrix<typename M::ElementType>::Identity(unitary.Rows());
        Matrix<typename M::ElementType> UUstar = unitary * unitary.ConjugateTransposed();

        return AreEqualMatrices(matrix, reconstruction) &&
               AreEqualMatrices(UUstar, identity) &&
               IsUpperTriangular(schur);
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurRealEigenvalues) {
        Matrix<double> A = {{1, 2, 3},
                            {0, 4, 5},
                            {0, 0, 6}};
        auto [U, S] = RealSchur(A);

        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurComplexEigenvalues) {
        Matrix<double> A = {{1, -1, 0},
                            {1, 1, 0},
                            {0, 0, 2}};
        auto [U, S] = RealSchur(A);

        EXPECT_TRUE(AreEqualMatrices(S, A));
        EXPECT_TRUE(IsOrthogonal(U));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurSymmetricMatrix) {
        Matrix<double> A = {{4, 1, 1},
                            {1, 4, 1},
                            {1, 1, 4}};
        auto [U, S] = RealSchur(A);

        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurAlreadyTriangular) {
        Matrix<double> A = {{1, 2, 3},
                            {0, 4, 5},
                            {0, 0, 6}};
        auto [U, S] = RealSchur(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(S == A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurZeroMatrix) {
        Matrix<double> A = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};
        auto [U, S] = RealSchur(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(U == expected_identity);
        EXPECT_TRUE(S == A);
        EXPECT_TRUE(CheckSchur(A, U, S));
}

TEST(TEST_SCHUR_DECOMPOSITION, RealSchurIllConditioned) {
        Matrix<double> A = {{1, 1e10, 0},
                            {0, 1, 1e10},
                            {0, 0, 1}};
        auto [U, S] = RealSchur(A);

        EXPECT_TRUE(CheckSchur(A, U, S));
}

