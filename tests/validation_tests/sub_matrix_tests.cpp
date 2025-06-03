#include "../../src/types/matrix.h"
#include "../../src/types/sub_matrix.h"

#include <complex>
#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace std::complex_literals;

TEST(TEST_SUBMATRIX, ConstructionAndElementAccess) {

        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        SubMatrix<double> sub(mat, {0, 1}, {0, 1});
        EXPECT_EQ(sub.Rows(), 2);
        EXPECT_EQ(sub.Columns(), 2);
        EXPECT_EQ(sub(0, 0), 1);
        EXPECT_EQ(sub(1, 1), 5);

        Matrix<double> rect_mat = {{1, 2, 3, 4},
                                   {5, 6, 7, 8}};
        SubMatrix<double> rect_sub(rect_mat, {0, 1}, {1, 2});
        EXPECT_EQ(rect_sub.Rows(), 2);
        EXPECT_EQ(rect_sub.Columns(), 2);
        EXPECT_EQ(rect_sub(0, 0), 2);
        EXPECT_EQ(rect_sub(1, 1), 7);

        Matrix<std::complex<double>> cmat = {{1.0 + 1.0i, 2.0 + 2.0i},
                                             {3.0 + 3.0i, 4.0 + 4.0i}};
        SubMatrix<std::complex<double>> csub(cmat);
        EXPECT_EQ(csub(1, 1), (4.0 + 4.0i));

        Matrix<double> small_mat = {{1e-10, 2e-10},
                                    {3e-10, 4e-10}};
        SubMatrix<double> small_sub(small_mat);
        EXPECT_DOUBLE_EQ(small_sub(0, 0), 1e-10);
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

        Matrix<double> rect_mat = {{1, 2, 3, 4},
                                   {5, 6, 7, 8},
                                   {9, 10, 11, 12}};
        SubMatrix<double> rect_sub(rect_mat, {0, 2}, {1, 3});
        auto rect_sub_sub = rect_sub.GetSubMatrix({1, 2}, {0, 1});
        Matrix<double> rect_expected = {{6, 7},
                                        {10, 11}};
        EXPECT_EQ(rect_sub_sub, rect_expected);
}

TEST(TEST_SUBMATRIX, RowAndColumnAccess) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        SubMatrix<double> sub(mat);

        auto row = sub.GetRow(1);
        EXPECT_EQ(row.Rows(), 1);
        EXPECT_EQ(row.Columns(), 3);
        EXPECT_EQ(row(0, 0), 4);
        EXPECT_EQ(row(0, 1), 5);
        EXPECT_EQ(row(0, 2), 6);

        auto col = sub.GetColumn(2);
        EXPECT_EQ(col.Rows(), 3);
        EXPECT_EQ(col.Columns(), 1);
        EXPECT_EQ(col(0, 0), 3);
        EXPECT_EQ(col(1, 0), 6);
        EXPECT_EQ(col(2, 0), 9);

        Matrix<double> single_row_mat = {{1, 2, 3}};
        SubMatrix<double> single_row_sub(single_row_mat);
        auto single_row = single_row_sub.GetRow(0);
        EXPECT_EQ(single_row.Rows(), 1);
        EXPECT_EQ(single_row.Columns(), 3);
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

        Matrix<double> small_mat = {{1e-10, 2e-10},
                                    {3e-10, 4e-10}};
        Matrix<double> small_add = {{1e-10, 2e-10},
                                    {3e-10, 4e-10}};
        SubMatrix<double> small_sub(small_mat);
        small_sub += small_add;
        Matrix<double> small_expected = {{2e-10, 4e-10},
                                         {6e-10, 8e-10}};
        EXPECT_TRUE(Core::AreEqualMatrices(small_mat, small_expected));
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
        EXPECT_EQ(mat(0, 1), 10);

        Matrix<std::complex<double>> cmat = {{1.0 + 1.0i, 2.0 + 2.0i},
                                             {3.0 + 3.0i, 4.0 + 4.0i}};
        SubMatrix<std::complex<double>> csub(cmat);
        csub.ConjugateTranspose();
        EXPECT_EQ(csub(0, 1), (3.0 - 3.0i));
}

TEST(TEST_SUBMATRIX, ElementwiseOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        SubMatrix<double> sub(mat);

        sub.Elementwise([](double& x) { x *= 2; });
        Matrix<double> expected = {{2, 4},
                                   {6, 8}};
        EXPECT_EQ(mat, expected);

        Matrix<double> small_mat = {{1e-10, 2e-10},
                                    {3e-10, 4e-10}};
        SubMatrix<double> small_sub(small_mat);
        small_sub.Elementwise([](double& x) { x *= 2; });
        EXPECT_TRUE(Core::AreEqualMatrices(small_mat, expected * 1e-10));
}

TEST(TEST_SUBMATRIX, EdgeCases) {

        Matrix<double> single = {{42}};
        SubMatrix<double> sub_single(single);
        EXPECT_EQ(sub_single.Rows(), 1);
        EXPECT_EQ(sub_single.Columns(), 1);
        EXPECT_EQ(sub_single(0, 0), 42);

        Matrix<double> row = {{1, 2, 3}};
        SubMatrix<double> sub_row(row);
        EXPECT_EQ(sub_row.Rows(), 1);
        EXPECT_EQ(sub_row.Columns(), 3);

        Matrix<double> col = {{1},
                              {2},
                              {3}};
        SubMatrix<double> sub_col(col);
        EXPECT_EQ(sub_col.Rows(), 3);
        EXPECT_EQ(sub_col.Columns(), 1);

        Matrix<double> small = {{1e-10, 2e-10},
                                {3e-10, 4e-10}};
        SubMatrix<double> sub_small(small);
        sub_small.RemoveZeros();
        Matrix<double> zero_matrix = {{0, 0},
                                      {0, 0}};
        EXPECT_TRUE(!Core::AreEqualMatrices(small, zero_matrix));
}

TEST(TEST_SUBMATRIX, MoveOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        SubMatrix<double> original(mat);
        SubMatrix<double> moved(std::move(original));

        EXPECT_EQ(moved.Rows(), 2);
        EXPECT_EQ(moved(0, 0), 1);

        SubMatrix<double> move_assigned = std::move(moved);
        EXPECT_EQ(move_assigned.Rows(), 2);
        EXPECT_EQ(move_assigned(0, 0), 1);
}

