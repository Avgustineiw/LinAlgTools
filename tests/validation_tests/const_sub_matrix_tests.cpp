#include "../../src/types/const_sub_matrix.h"
#include "../../src/types/matrix.h"

#include <gtest/gtest.h>

using namespace LinAlgTools;

TEST(TEST_CONST_SUBMATRIX, ConstructionAndAccess) {
        Matrix<double> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        ConstSubMatrix<double> csub(mat);

        EXPECT_EQ(csub.Rows(), 3);
        EXPECT_EQ(csub.Columns(), 3);
        EXPECT_EQ(csub(0, 0), 1);
        EXPECT_EQ(csub(1, 1), 5);
}

TEST(TEST_CONST_SUBMATRIX, Operations) {
        Matrix<double> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        ConstSubMatrix<double> csub(mat);

        EXPECT_EQ(csub.Trace(), 15);
        EXPECT_EQ(csub.Diagonal(), Matrix<double>({{1}, {5}, {9}}));

        Matrix<double> vecMat = {{1}, {2}, {3}};
        ConstSubMatrix<double> vecSub(vecMat);
        EXPECT_DOUBLE_EQ(vecSub.Get2Norm(), std::sqrt(14));
}

TEST(TEST_CONST_SUBMATRIX, Slicing) {
        Matrix<double> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        ConstSubMatrix<double> csub(mat, {0, 1}, {0, 1});

        ConstSubMatrix<double> sub = csub.GetConstSubMatrix({1, 1}, {1, 1});
        EXPECT_EQ(sub(0, 0), 5);
}

TEST(TEST_CONST_SUBMATRIX, RowAndColumnAccess) {
        Matrix<double> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        ConstSubMatrix<double> csub(mat);

        auto row = csub.GetRow(2);
        EXPECT_EQ(row(0, 0), 4);

        auto col = csub.GetColumn(3);
        EXPECT_EQ(col(2, 0), 9);
}

TEST(TEST_CONST_SUBMATRIX, TransposedView) {
        Matrix<double> mat = {{1, 2, 3}, {4, 5, 6}};
        ConstSubMatrix<double> csub(mat);
        auto transposed = csub.Transposed();

        EXPECT_EQ(transposed(2, 1), 6);
}

