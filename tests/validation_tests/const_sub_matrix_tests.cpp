#include "../../src/types/const_sub_matrix.h"
#include "../../src/types/matrix.h"

#include <complex>
#include <gtest/gtest.h>

using namespace LinAlgTools;
using namespace std::complex_literals;

TEST(TEST_CONST_SUBMATRIX, ConstructionAndAccess) {

        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        ConstSubMatrix<double> sub(mat, {0, 1}, {0, 1});
        EXPECT_EQ(sub.Rows(), 2);
        EXPECT_EQ(sub.Columns(), 2);
        EXPECT_EQ(sub(0, 0), 1);
        EXPECT_EQ(sub(1, 1), 5);

        Matrix<double> rect_mat = {{1, 2, 3, 4},
                                   {5, 6, 7, 8}};
        ConstSubMatrix<double> rect_sub(rect_mat, {0, 1}, {1, 2});
        EXPECT_EQ(rect_sub.Rows(), 2);
        EXPECT_EQ(rect_sub.Columns(), 2);
        EXPECT_EQ(rect_sub(0, 0), 2);
        EXPECT_EQ(rect_sub(1, 1), 7);

        Matrix<std::complex<double>> cmat = {{1.0 + 1.0i, 2.0 + 2.0i},
                                             {3.0 + 3.0i, 4.0 + 4.0i}};
        ConstSubMatrix<std::complex<double>> csub(cmat);
        EXPECT_EQ(csub(1, 1), (4.0 + 4.0i));

        Matrix<double> small_mat = {{1e-10, 2e-10},
                                    {3e-10, 4e-10}};
        ConstSubMatrix<double> small_sub(small_mat);
        EXPECT_DOUBLE_EQ(small_sub(0, 0), 1e-10);
}

TEST(TEST_CONST_SUBMATRIX, SubMatrixOperations) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        ConstSubMatrix<double> sub(mat);

        auto sub_sub = sub.GetSubMatrix({1, 2}, {0, 1});
        Matrix<double> expected = {{4, 5},
                                   {7, 8}};
        EXPECT_EQ(sub_sub, expected);

        auto full_sub = sub.GetSubMatrix({0, 2}, {0, 2});
        EXPECT_EQ(full_sub, mat);
}

TEST(TEST_CONST_SUBMATRIX, ConstSubMatrixOperations) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        SubMatrix<double> sub(mat);
        ConstSubMatrix<double> csub = sub.Transpose();
        sub *= 5;
        EXPECT_EQ(mat, csub);
}


TEST(TEST_CONST_SUBMATRIX, RowAndColumnAccess) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}};
        ConstSubMatrix<double> sub(mat);

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

        Matrix<double> single = {{42}};
        ConstSubMatrix<double> sub_single(single);
        auto single_row = sub_single.GetRow(0);
        EXPECT_EQ(single_row.Rows(), 1);
        EXPECT_EQ(single_row.Columns(), 1);
        EXPECT_EQ(single_row(0, 0), 42);
}

TEST(TEST_CONST_SUBMATRIX, MatrixOperations) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6}};
        ConstSubMatrix<double> sub(mat);

        auto diag = sub.Diagonal();
        Matrix<double> expected_diag = {{1},
                                        {5}};
        EXPECT_EQ(diag, expected_diag);

        EXPECT_EQ(sub.Trace(), 6);

        Matrix<double> small_mat = {{1e-10, 0, 0},
                                    {0, 2e-10, 0},
                                    {0, 0, 3e-10}};
        ConstSubMatrix<double> small_sub(small_mat);
        EXPECT_DOUBLE_EQ(small_sub.Trace(), 6e-10);
}

TEST(TEST_CONST_SUBMATRIX, NormCalculations) {

        Matrix<double> vec = {{1},
                              {2},
                              {3}};
        ConstSubMatrix<double> sub_vec(vec);
        EXPECT_DOUBLE_EQ(sub_vec.GetVector2Norm(), std::sqrt(14));

        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        ConstSubMatrix<double> sub_mat(mat);
        EXPECT_DOUBLE_EQ(sub_mat.GetFrobeniusNorm(), std::sqrt(30));

        Matrix<std::complex<double>> cvec = {{1.0 + 1.0i},
                                             {2.0 + 2.0i}};
        ConstSubMatrix<std::complex<double>> csub_vec(cvec);
        EXPECT_TRUE(Core::IsZero(csub_vec.GetVector2Norm() - std::sqrt(10)));

        Matrix<double> small_vec = {{1e-10},
                                    {2e-10}};
        ConstSubMatrix<double> small_sub_vec(small_vec);
        EXPECT_DOUBLE_EQ(small_sub_vec.GetVector2Norm(), std::sqrt(5e-20));
}

TEST(TEST_CONST_SUBMATRIX, TransposeOperations) {
        Matrix<double> mat = {{1, 2, 3},
                              {4, 5, 6}};
        ConstSubMatrix<double> sub(mat);

        auto transposed = Transposed(sub);
        Matrix<double> expected = {{1, 4},
                                   {2, 5},
                                   {3, 6}};
        EXPECT_EQ(transposed, expected);

        Matrix<std::complex<double>> cmat = {{1.0 + 1.0i, 2.0 + 2.0i},
                                             {3.0 + 3.0i, 4.0 + 4.0i}};
        ConstSubMatrix<std::complex<double>> csub(cmat);
        auto conj_transposed = ConjugateTransposed(csub);
        Matrix<std::complex<double>> expected_ct = {{1.0 - 1.0i, 3.0 - 3.0i},
                                                    {2.0 - 2.0i, 4.0 - 4.0i}};
        EXPECT_EQ(conj_transposed, expected_ct);
}

TEST(TEST_CONST_SUBMATRIX, ElementwiseOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        ConstSubMatrix<double> sub(mat);

        double sum = 0;
        sub.Elementwise([&sum](const double& x) { sum += x; });
        EXPECT_EQ(sum, 10);

        Matrix<std::complex<double>> cmat = {{1.0 + 1.0i, 2.0 + 2.0i},
                                             {3.0 + 3.0i, 4.0 + 4.0i}};
        ConstSubMatrix<std::complex<double>> csub(cmat);
        std::complex<double> csum = 0;
        csub.Elementwise([&csum](const std::complex<double>& x) { csum += x; });
        EXPECT_EQ(csum, (10.0 + 10.0i));
}

TEST(TEST_CONST_SUBMATRIX, EdgeCases) {

        Matrix<double> single = {{42}};
        ConstSubMatrix<double> sub_single(single);
        EXPECT_EQ(sub_single.Rows(), 1);
        EXPECT_EQ(sub_single.Columns(), 1);
        EXPECT_EQ(sub_single(0, 0), 42);

        Matrix<double> row = {{1, 2, 3}};
        ConstSubMatrix<double> sub_row(row);
        EXPECT_EQ(sub_row.Rows(), 1);
        EXPECT_EQ(sub_row.Columns(), 3);

        Matrix<double> col = {{1},
                              {2},
                              {3}};
        ConstSubMatrix<double> sub_col(col);
        EXPECT_EQ(sub_col.Rows(), 3);
        EXPECT_EQ(sub_col.Columns(), 1);

        Matrix<double> small = {{1e-10, 2e-10},
                                {3e-10, 4e-10}};
        ConstSubMatrix<double> sub_small(small);
        auto small_diag = sub_small.Diagonal();
        EXPECT_DOUBLE_EQ(small_diag(0, 0), 1e-10);
        EXPECT_DOUBLE_EQ(small_diag(1, 0), 4e-10);
}

TEST(TEST_CONST_SUBMATRIX, MoveOperations) {
        Matrix<double> mat = {{1, 2},
                              {3, 4}};
        ConstSubMatrix<double> original(mat);
        ConstSubMatrix<double> moved(std::move(original));

        EXPECT_EQ(moved.Rows(), 2);
        EXPECT_EQ(moved(0, 0), 1);

        ConstSubMatrix<double> move_assigned = std::move(moved);
        EXPECT_EQ(move_assigned.Rows(), 2);
        EXPECT_EQ(move_assigned(0, 0), 1);
}

