#include "../../src/types/matrix.h"
#include "../../src/types/sub_matrix.h"

#include <gtest/gtest.h>

using namespace LinAlgTools;

TEST(TEST_SUBMATRIX, ConstructionAndElementAccess) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        SubMatrix<double> sub(mat, {0, 1}, {0, 1});

        EXPECT_EQ(sub.Rows(), 2);
        EXPECT_EQ(sub.Columns(), 2);
        EXPECT_EQ(sub(0, 0), 1);
        EXPECT_EQ(sub(1, 1), 5);

        sub(1, 1) = 10;
        EXPECT_EQ(mat(1, 1), 10);
}

TEST(TEST_SUBMATRIX, GetSubMatrix) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        SubMatrix<double> sub(mat, {0, 2}, {0, 2});
        auto subSub = sub.GetSubMatrix({1, 2}, {0, 1});

        Matrix<double> expected = {{4, 5},
                                   {7, 8}};
        EXPECT_EQ(subSub, expected);
}

TEST(TEST_SUBMATRIX, RowAndColumnAccess) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        SubMatrix<double> sub(mat);

        auto row = sub.GetRow(2);
        EXPECT_EQ(row.Rows(), 1);
        EXPECT_EQ(row.Columns(), 3);
        EXPECT_EQ(row(0, 0), 4);
        EXPECT_EQ(row(0, 1), 5);
        EXPECT_EQ(row(0, 2), 6);

        auto col = sub.GetColumn(3);
        EXPECT_EQ(col.Rows(), 3);
        EXPECT_EQ(col.Columns(), 1);
        EXPECT_EQ(col(0, 0), 3);
        EXPECT_EQ(col(1, 0), 6);
        EXPECT_EQ(col(2, 0), 9);
}

TEST(TEST_SUBMATRIX, InPlaceOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        SubMatrix<double> sub(mat);
        Matrix<double> add = {{5, 6},
                              {7, 8}};

        sub += add;
        Matrix<double> expected = {{6, 8},
                                   {10, 12}};
        EXPECT_EQ(mat, expected);
}

TEST(TEST_SUBMATRIX, Transpose) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6}};
        SubMatrix<double> sub(mat);

        sub.Transpose();
        EXPECT_EQ(sub.Rows(), 3);
        EXPECT_EQ(sub.Columns(), 2);
        EXPECT_EQ(sub(0, 0), 1);
        EXPECT_EQ(sub(0, 1), 4);
        EXPECT_EQ(sub(1, 0), 2);
        EXPECT_EQ(sub(1, 1), 5);
        EXPECT_EQ(sub(2, 0), 3);
        EXPECT_EQ(sub(2, 1), 6);

        sub(0, 1) = 10;
        EXPECT_EQ(mat(1, 0), 10);
}

TEST(TEST_SUBMATRIX, ElementwiseOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        SubMatrix<double> sub(mat);

        sub.Elementwise([](double& x) { x *= 2; });
        Matrix<double> expected = {{2, 4},
                                   {6, 8}};
        EXPECT_EQ(mat, expected);
}

TEST(TEST_SUBMATRIX, MoveOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        SubMatrix<double> original(mat);
        SubMatrix<double> moved(std::move(original));

        EXPECT_EQ(moved.Rows(), 2);
        EXPECT_EQ(moved(0, 0), 1);
}

