#include "../../src/algorithms/qr_decomposition.h"
#include "../../src/core/math_utils.h"
#include "../../src/core/matrix_traits.h"
#include "../../src/types/matrix.h"

#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace LinAlgTools::Core;
using namespace LinAlgTools::Algorithm;

template<typename T = double>
using Index = Core::Indices::Index;

template<MatrixType M, MatrixType F, MatrixType S>
bool CheckQR(const M& matrix, const F& Q, const S& R) {
        return (AreEqualMatrices(matrix, Q * R) &&
                IsOrthogonal(Q) &&
                IsUpperTriangular(R));
}

TEST(TEST_QR_DECOMPOSITION, HouseholderSquareMatrix) {
        Matrix<double> A = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
        auto [Q, R] = LinAlgTools::Algorithm::HouseholderQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, HouseholderRectangularMatrix) {
        Matrix<double> A = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        auto [Q, R] = LinAlgTools::Algorithm::HouseholderQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, HouseholderSingleColumn) {
        Matrix<double> A = {{1}, {2}, {3}, {4}};
        auto [Q, R] = LinAlgTools::Algorithm::HouseholderQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, HouseholderSingleRow) {
        Matrix<double> A = {{1, 2, 3, 4, 5}};
        auto [Q, R] = LinAlgTools::Algorithm::HouseholderQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}


TEST(TEST_QR_DECOMPOSITION, HouseholderZeroMatrix) {
        Matrix<double> A = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        auto [Q, R] = LinAlgTools::Algorithm::HouseholderQR(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(Q == expected_identity) << "Q should be identity for zero matrix";

        Matrix<double> zero_matrix(A.Rows(), A.Columns());
        EXPECT_TRUE(R == zero_matrix) << "R should be zero for zero matrix";

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, HouseholderAlreadyUpperTriangular) {
        Matrix<double> A = {{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
        auto [Q, R] = LinAlgTools::Algorithm::HouseholderQR(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(Q == expected_identity) << "Q should be identity for upper triangular matrix";

        EXPECT_TRUE(R == A) << "R should equal original matrix when it's already upper triangular";

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, GivensSquareMatrix) {
        Matrix<double> A = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
        auto [Q, R] = LinAlgTools::Algorithm::GivensQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, GivensRectangularMatrix) {
        Matrix<double> A = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        auto [Q, R] = LinAlgTools::Algorithm::GivensQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, GivensSingleColumn) {
        Matrix<double> A = {{1}, {2}, {3}, {4}};
        auto [Q, R] = LinAlgTools::Algorithm::GivensQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, GivensSingleRow) {
        Matrix<double> A = {{1, 2, 3, 4, 5}};
        auto [Q, R] = LinAlgTools::Algorithm::GivensQR(A);

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}


TEST(TEST_QR_DECOMPOSITION, GivensZeroMatrix) {
        Matrix<double> A = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        auto [Q, R] = LinAlgTools::Algorithm::GivensQR(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(Q == expected_identity) << "Q should be identity for zero matrix";

        Matrix<double> zero_matrix(A.Rows(), A.Columns());
        EXPECT_TRUE(R == zero_matrix) << "R should be zero for zero matrix";

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

TEST(TEST_QR_DECOMPOSITION, GivensAlreadyUpperTriangular) {
        Matrix<double> A = {{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
        auto [Q, R] = LinAlgTools::Algorithm::GivensQR(A);

        Matrix<double> expected_identity = Matrix<double>::Identity(A.Rows());
        EXPECT_TRUE(Q == expected_identity) << "Q should be identity for upper triangular matrix";

        EXPECT_TRUE(R == A) << "R should equal original matrix when it's already upper triangular";

        EXPECT_TRUE(CheckQR(A, Q, R)) << "QR does not reconstruct original matrix.";
}

