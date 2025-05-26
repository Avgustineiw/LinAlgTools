#include "../../src/types/matrix.h"

#include <complex>
#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace std::complex_literals;

TEST(TEST_MATRIX, ConstructionAndBasicProperties) {
        Matrix<double> m1(3);
        EXPECT_EQ(m1.Rows(), 3);
        EXPECT_EQ(m1.Columns(), 3);

        Matrix<double> m2(3, 4);
        EXPECT_EQ(m2.Rows(), 3);
        EXPECT_EQ(m2.Columns(), 4);

        Matrix<double> m3 = {{1, 2, 3},
                             {4, 5, 6}};
        EXPECT_EQ(m3.Rows(), 2);
        EXPECT_EQ(m3.Columns(), 3);

        Matrix<std::complex<double>> m4 = {{1.0 + 2.0i, 3.0 - 1.0i},
                                           {4.0, 5.0 + 0.5i}};
        EXPECT_EQ(m4.Rows(), 2);
        EXPECT_EQ(m4.Columns(), 2);
}

TEST(TEST_MATRIX, ElementAccess) {
        Matrix<double> m = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9}};
        EXPECT_EQ(m(0, 0), 1);
        EXPECT_EQ(m(1, 1), 5);
        EXPECT_EQ(m(2, 2), 9);

        m(1, 1) = 10;
        EXPECT_EQ(m(1, 1), 10);

        Matrix<double> small_m = {{1e-10, 2e-10},
                                  {3e-10, 4e-10}};
        EXPECT_DOUBLE_EQ(small_m(0, 0), 1e-10);
        small_m(0, 0) = 5e-10;
        EXPECT_DOUBLE_EQ(small_m(0, 0), 5e-10);
}

TEST(TEST_MATRIX, MatrixOperations) {
        Matrix<double> a = {{1, 2},
                            {3, 4}};
        Matrix<double> b = {{5, 6},
                            {7, 8}};

        Matrix<double> c = {{1, 2, 3},
                            {4, 5, 6}};
        Matrix<double> d = {{7, 8},
                            {9, 10},
                            {11, 12}};

        Matrix<double> sum = a + b;
        Matrix<double> expectedSum = {{6, 8},
                                      {10, 12}};
        EXPECT_EQ(sum, expectedSum);

        Matrix<double> product = c * d;
        Matrix<double> expectedProduct = {{58, 64},
                                          {139, 154}};
        EXPECT_EQ(product, expectedProduct);

        Matrix<double> smallA = {{1e-10, 2e-10},
                                 {3e-10, 4e-10}};
        Matrix<double> smallB = {{2e-10, 3e-10},
                                 {4e-10, 5e-10}};
        Matrix<double> smallSum = smallA + smallB;
        Matrix<double> expectedSmallSum = {{3e-10, 5e-10},
                                           {7e-10, 9e-10}};
        for (Index i = 0; i < smallSum.Rows(); ++i) {
                for (Index j = 0; j < smallSum.Columns(); ++j) {
                        EXPECT_DOUBLE_EQ(smallSum(i, j), expectedSmallSum(i, j));
                }
        }

        Matrix<std::complex<double>> ca = {{1.0 + 1.0i, 2.0},
                                           {3.0, 4.0 - 1.0i}};
        Matrix<std::complex<double>> cb = {{5.0, 6.0 + 1.0i},
                                           {7.0 - 2.0i, 8.0}};
        Matrix<std::complex<double>> csum = ca + cb;
        Matrix<std::complex<double>> expectedCsum = {{6.0 + 1.0i, 8.0 + 1.0i},
                                                     {10.0 - 2.0i, 12.0 - 1.0i}};
        EXPECT_EQ(csum, expectedCsum);
}

TEST(TEST_MATRIX, TransposeOperations) {
        Matrix<double> m1 = {{1, 2},
                             {3, 4}};
        Matrix<double> transposed1 = m1.Transposed();
        Matrix<double> expected1 = {{1, 3},
                                    {2, 4}};
        EXPECT_EQ(transposed1, expected1);

        Matrix<double> m2 = {{1, 2, 3},
                             {4, 5, 6}};
        Matrix<double> transposed2 = m2.Transposed();
        Matrix<double> expected2 = {{1, 4},
                                    {2, 5},
                                    {3, 6}};
        EXPECT_EQ(transposed2, expected2);

        Matrix<std::complex<double>> m3 = {{1.0 + 2.0i, 3.0 - 1.0i},
                                           {4.0i, -2.0 - 3.0i}};
        Matrix<std::complex<double>> conjTransposed = m3.ConjugateTransposed();
        Matrix<std::complex<double>> expected3 = {{1.0 - 2.0i, -4.0i},
                                                  {3.0 + 1.0i, -2.0 + 3.0i}};
        EXPECT_EQ(conjTransposed, expected3);
}

TEST(TEST_MATRIX, EdgeCases) {
        Matrix<double> single = {{42}};
        EXPECT_EQ(single.Rows(), 1);
        EXPECT_EQ(single.Columns(), 1);
        EXPECT_EQ(single(0, 0), 42);

        Matrix<double> row = {{1, 2, 3}};
        EXPECT_EQ(row.Rows(), 1);
        EXPECT_EQ(row.Columns(), 3);

        Matrix<double> col = {{1},
                              {2},
                              {3}};
        EXPECT_EQ(col.Rows(), 3);
        EXPECT_EQ(col.Columns(), 1);

        Matrix<double> small = {{1e-10, 2e-10},
                                {3e-10, 4e-10}};
        auto smallTransposed = small.Transposed();
        Matrix<double> expectedSmallTransposed = {{1e-10, 3e-10},
                                                  {2e-10, 4e-10}};
        EXPECT_EQ(smallTransposed, expectedSmallTransposed);

        Matrix<std::complex<double>> complexSingle = {{1e-10 + 2e-10i}};
        EXPECT_EQ(complexSingle.Rows(), 1);
        EXPECT_EQ(complexSingle.Columns(), 1);
        EXPECT_EQ(complexSingle(0, 0), std::complex<double>(1e-10, 2e-10));
}

TEST(TEST_MATRIX, MixedTypeOperations) {
        Matrix<double> md = {{1, 2},
                             {3, 4}};
        Matrix<float> mf = {{5, 6},
                            {7, 8}};

        auto result = md * mf;
        Matrix<double> expected = {{19, 22},
                                   {43, 50}};
        EXPECT_EQ(result, expected);

        Matrix<std::complex<double>> mc = {{1.0 + 1.0i, 2.0},
                                           {3.0, 4.0 - 1.0i}};
        auto result2 = mc * md;
        Matrix<std::complex<double>> expected2 = {{7.0 + 1.0i, 10.0 + 2.0i},
                                                  {15.0 - 3.0i, 22.0 - 4.0i}};
        EXPECT_EQ(result2, expected2);

        Matrix<double> smallA = {{1e-10, 2e-10},
                                 {3e-10, 4e-10}};
        Matrix<double> smallB = {{2, 3},
                                 {4, 5}};
        auto smallResult = smallA * smallB;
        Matrix<double> expectedSmallResult = {{1e-9, 1.3e-9},
                                              {2.2e-9, 2.9e-9}};
        for (Index i = 0; i < smallResult.Rows(); ++i) {
                for (Index j = 0; j < smallResult.Columns(); ++j) {
                        EXPECT_DOUBLE_EQ(smallResult(i, j), expectedSmallResult(i, j));
                }
        }
}

TEST(TEST_MATRIX, CopyAndMoveOperations) {
        Matrix<double> original = {{1, 2},
                                   {3, 4}};

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

TEST(TEST_MATRIX, InPlaceOperations) {
        Matrix<double> a = {{1, 2},
                            {3, 4}};
        Matrix<double> b = {{5, 6},
                            {7, 8}};

        Matrix<double> a_plus_b = a;
        a_plus_b += b;
        EXPECT_EQ(a_plus_b, (Matrix<double>{{6, 8},
                                            {10, 12}}));

        Matrix<double> a_minus_b = a;
        a_minus_b -= b;
        EXPECT_EQ(a_minus_b, (Matrix<double>{{-4, -4},
                                             {-4, -4}}));

        Matrix<double> a_times_b = a;
        a_times_b *= b;
        EXPECT_EQ(a_times_b, (Matrix<double>{{19, 22},
                                             {43, 50}}));

        Matrix<double> a_scaled = a;
        a_scaled *= 2.0;
        EXPECT_EQ(a_scaled, (Matrix<double>{{2, 4},
                                            {6, 8}}));
}

TEST(TEST_MATRIX, SpecialMatrices) {
        Matrix<double> identity = Matrix<double>::Identity(3);
        Matrix<double> expectedIdentity = {{1, 0, 0},
                                           {0, 1, 0},
                                           {0, 0, 1}};
        EXPECT_EQ(identity, expectedIdentity);

        Matrix<double> m = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9}};
        Matrix<double> diagonal = m.Diagonal();
        Matrix<double> expectedDiagonal = {{1},
                                           {5},
                                           {9}};
        EXPECT_EQ(diagonal, expectedDiagonal);

        EXPECT_EQ(m.Trace(), 15);
}

TEST(TEST_MATRIX, SubMatrixOperations) {
        Matrix<double> m = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9}};

        auto row = m.GetRow(1);
        Matrix<double> expectedRow = {{4, 5, 6}};
        EXPECT_EQ(row, expectedRow);

        auto col = m.GetColumn(2);
        Matrix<double> expectedCol = {{3},
                                      {6},
                                      {9}};
        EXPECT_EQ(col, expectedCol);

        auto sub = m.GetSubMatrix({1, 2},
                                  {0, 1});
        Matrix<double> expectedSub = {{4, 5},
                                      {7, 8}};
        EXPECT_EQ(sub, expectedSub);
}

TEST(TEST_MATRIX, NormAndNormalization) {
        Matrix<double> m = {{1},
                            {2},
                            {3},
                            {4}};

        double norm = m.GetVector2Norm();
        EXPECT_DOUBLE_EQ(norm, std::sqrt(1 + 4 + 9 + 16));

        Matrix<double> normalized = m;
        normalized.Normalize();
        double normalizedNorm = normalized.GetVector2Norm();
        EXPECT_NEAR(normalizedNorm, 1.0, 1e-10);
}

TEST(TEST_MATRIX, ElementwiseOperations) {
        Matrix<double> m = {{1, 2},
                            {3, 4}};

        m.Elementwise([](double& x) { x = x * x; });
        Matrix<double> expected = {{1, 4},
                                   {9, 16}};
        EXPECT_EQ(m, expected);
}

TEST(TEST_MATRIX, ComparisonOperators) {
        Matrix<double> m1 = {{1, 2},
                             {3, 4}};
        Matrix<double> m2 = {{1, 2},
                             {3, 4}};
        Matrix<double> m3 = {{5, 6},
                             {7, 8}};

        EXPECT_TRUE(m1 == m2);
        EXPECT_FALSE(m1 == m3);

        EXPECT_FALSE(m1 != m2);
        EXPECT_TRUE(m1 != m3);

        Matrix<double> m4 = {{1, 2, 3},
                             {4, 5, 6}};
        EXPECT_FALSE(m1 == m4);
        EXPECT_TRUE(m1 != m4);
}

