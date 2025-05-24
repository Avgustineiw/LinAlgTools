#include "../../src/types/matrix.h"

#include <gtest/gtest.h>

using namespace LinAlgTools;

TEST(TEST_MATRIX, ConstructionAndBasicProperties) {

        Matrix<double> m1(3, 4);
        EXPECT_EQ(m1.Rows(), 3);
        EXPECT_EQ(m1.Columns(), 4);

        Matrix<double> m2 = {{1, 2, 3}, {4, 5, 6}};
        EXPECT_EQ(m2.Rows(), 2);
        EXPECT_EQ(m2.Columns(), 3);

        Matrix<double> m3(5);
        EXPECT_EQ(m3.Rows(), 5);
        EXPECT_EQ(m3.Columns(), 5);
}

TEST(TEST_MATRIX, ElementAccess) {
        Matrix<double> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        EXPECT_EQ(m(0, 0), 1);
        EXPECT_EQ(m(1, 1), 5);
        EXPECT_EQ(m(2, 2), 9);

        m(1, 1) = 10;
        EXPECT_EQ(m(1, 1), 10);
}

TEST(TEST_MATRIX, CopyAndMoveOperations) {
        Matrix<double> original = {{1, 2}, {3, 4}};

        Matrix<double> copy(original);
        EXPECT_EQ(copy, original);

        Matrix<double> moved(std::move(copy));
        EXPECT_EQ(moved, original);
        EXPECT_EQ(copy.Rows(), 0);

        Matrix<double> copyAssigned = original;
        EXPECT_EQ(copyAssigned, original);

        Matrix<double> moveAssigned = std::move(copyAssigned);
        EXPECT_EQ(moveAssigned, original);
        EXPECT_EQ(copyAssigned.Rows(), 0);
}

TEST(TEST_MATRIX, MatrixOperations) {
        Matrix<double> a = {{1, 2}, {3, 4}};
        Matrix<double> b = {{5, 6}, {7, 8}};

        Matrix<double> sum = a + b;
        Matrix<double> expectedSum = {{6, 8}, {10, 12}};
        EXPECT_EQ(sum, expectedSum);

        Matrix<double> diff = a - b;
        Matrix<double> expectedDiff = {{-4, -4}, {-4, -4}};
        EXPECT_EQ(diff, expectedDiff);

        Matrix<double> product = a * b;
        Matrix<double> expectedProduct = {{19, 22}, {43, 50}};
        EXPECT_EQ(product, expectedProduct);

        Matrix<double> scaled = a * 2.0;
        Matrix<double> expectedScaled = {{2, 4}, {6, 8}};
        EXPECT_EQ(scaled, expectedScaled);
}

TEST(TEST_MATRIX, InPlaceOperations) {
        Matrix<double> a = {{1, 2}, {3, 4}};
        Matrix<double> b = {{5, 6}, {7, 8}};

        Matrix<double> a_plus_b = a;
        a_plus_b += b;
        EXPECT_EQ(a_plus_b, (Matrix<double>{{6, 8}, {10, 12}}));

        Matrix<double> a_minus_b = a;
        a_minus_b -= b;
        EXPECT_EQ(a_minus_b, (Matrix<double>{{-4, -4}, {-4, -4}}));

        Matrix<double> a_times_b = a;
        a_times_b *= b;
        EXPECT_EQ(a_times_b, (Matrix<double>{{19, 22}, {43, 50}}));

        Matrix<double> a_scaled = a;
        a_scaled *= 2.0;
        EXPECT_EQ(a_scaled, (Matrix<double>{{2, 4}, {6, 8}}));
}

TEST(TEST_MATRIX, TransposeOperations) {
        Matrix<double> m = {{1, 2, 3}, {4, 5, 6}};

        Matrix<double> transposed = m.Transposed();
        Matrix<double> expectedTransposed = {{1, 4}, {2, 5}, {3, 6}};
        EXPECT_EQ(transposed, expectedTransposed);

        m.Transpose();
        EXPECT_EQ(m, expectedTransposed);
}

TEST(TEST_MATRIX, SpecialMatrices) {

        Matrix<double> identity = Matrix<double>::Identity(3);
        Matrix<double> expectedIdentity = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        EXPECT_EQ(identity, expectedIdentity);

        Matrix<double> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        Matrix<double> diagonal = m.Diagonal();
        Matrix<double> expectedDiagonal = {{1}, {5}, {9}};
        EXPECT_EQ(diagonal, expectedDiagonal);

        EXPECT_EQ(m.Trace(), 15);
}

TEST(TEST_MATRIX, SubMatrixOperations) {
        Matrix<double> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        auto row = m.GetRow(2);
        Matrix<double> expectedRow = {{4, 5, 6}};
        EXPECT_EQ(row, expectedRow);

        auto col = m.GetColumn(3);
        Matrix<double> expectedCol = {{3}, {6}, {9}};
        EXPECT_EQ(col, expectedCol);

        auto sub = m.GetSubMatrix({1, 2}, {0, 1});
        Matrix<double> expectedSub = {{4, 5}, {7, 8}};
        EXPECT_EQ(sub, expectedSub);
}

TEST(TEST_MATRIX, NormAndNormalization) {
        Matrix<double> m = {{1}, {2}, {3}, {4}};

        double norm = m.GetVector2Norm();
        EXPECT_DOUBLE_EQ(norm, std::sqrt(1 + 4 + 9 + 16));

        Matrix<double> normalized = m;
        normalized.Normalize();
        double normalizedNorm = normalized.GetVector2Norm();
        EXPECT_NEAR(normalizedNorm, 1.0, 1e-10);
}

TEST(TEST_MATRIX, ElementwiseOperations) {
        Matrix<double> m = {{1, 2}, {3, 4}};

        m.Elementwise([](double& x) { x = x * x; });
        Matrix<double> expected = {{1, 4}, {9, 16}};
        EXPECT_EQ(m, expected);
}

TEST(TEST_MATRIX, ComparisonOperators) {
        Matrix<double> m1 = {{1, 2}, {3, 4}};
        Matrix<double> m2 = {{1, 2}, {3, 4}};
        Matrix<double> m3 = {{5, 6}, {7, 8}};

        EXPECT_TRUE(m1 == m2);
        EXPECT_FALSE(m1 == m3);

        EXPECT_FALSE(m1 != m2);
        EXPECT_TRUE(m1 != m3);

        Matrix<double> m4 = {{1, 2, 3}, {4, 5, 6}};
        EXPECT_FALSE(m1 == m4);
        EXPECT_TRUE(m1 != m4);
}

TEST(TEST_MATRIX, EdgeCases) {

        Matrix<double> single = {{42}};
        EXPECT_EQ(single.Rows(), 1);
        EXPECT_EQ(single.Columns(), 1);
        EXPECT_EQ(single(0, 0), 42);

        Matrix<double> row = {{1, 2, 3}};
        EXPECT_EQ(row.Rows(), 1);
        EXPECT_EQ(row.Columns(), 3);

        Matrix<double> col = {{1}, {2}, {3}};
        EXPECT_EQ(col.Rows(), 3);
        EXPECT_EQ(col.Columns(), 1);
}

