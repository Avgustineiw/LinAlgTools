#include "../src/types/matrix.h"
#include <gtest/gtest.h>

template<typename T = double>
using Matrix = LinAlgTools::Matrix<T>;

TEST(TEST_MATRIX, BasicConstructors) {
        {
                Matrix<float> square(5, 5);
                EXPECT_EQ(square.Rows(), 5);
                EXPECT_EQ(square.Columns(), 5);

                Matrix<double> rect(2, 3);
                EXPECT_EQ(rect.Rows(), 2);
                EXPECT_EQ(rect.Columns(), 3);

                EXPECT_TRUE(rect == Matrix<double>({{0, 0, 0}, {0, 0, 0}}));
        }
        {
                Matrix<long double> matrix = {{1, 2, 3}, {4, 5, 6}};
                EXPECT_TRUE(matrix == Matrix<long double>({{1, 2, 3}, {4, 5, 6}}));
        }
}

TEST(TEST_MATRIX, CopySemantics) {
        using Matrix = Matrix<double>;
        Matrix m1(2, 2);

        {
                Matrix m2 = {{1.0, 7.4}, {4.1, 5.6}};
                m1 = m2;
                EXPECT_TRUE(m1 == m2);

                m2(0, 0) = 0.3;
                EXPECT_FALSE(m1 == m2);
        }

        EXPECT_TRUE(m1 == Matrix({{1.0, 7.4}, {4.1, 5.6}}));
}

TEST(TEST_MATRIX, MoveSemantics) {
        using Matrix = Matrix<float>;
        Matrix m1(2, 2);

        {
                Matrix m2 = {{1, -1}, {0, 2}, {-1, 0}, {-2, 1}};
                m1 = std::move(m2);
        }

        EXPECT_TRUE(m1 == Matrix({{1, -1}, {0, 2}, {-1, 0}, {-2, 1}}));
}

void CheckArithmeticSum() {
        using Matrix = Matrix<double>;

        Matrix m1 = {{1, 2, 3}, {4, 5, 6}};
        Matrix m2 = {{7, 8, 9}, {10, 11, 12}};
        EXPECT_TRUE(m1 + m2 == Matrix({{8, 10, 12}, {14, 16, 18}}));

        m1 += m1;
        m1 += m2;
        EXPECT_TRUE(m1 == Matrix({{9, 12, 15}, {18, 21, 24}}));
}

void CheckArithmeticDiff() {
        using Matrix = Matrix<double>;

        Matrix m1 = {{9, 4}, {5, 1}, {12, 9}};
        Matrix m2 = {{-3, 0}, {1, 4}, {6, -12}};
        EXPECT_TRUE(m1 - m2 == Matrix({{12, 4}, {4, -3}, {6, 21}}));

        m1 -= m2;
        m1 -= m2;
        EXPECT_TRUE(m1 == Matrix({{15, 4}, {3, -7}, {0, 33}}));
}

void CheckArithmeticMulti() {
        using Matrix = Matrix<double>;

        Matrix m1 = {{8, 6, 1}, {8, 5, 1}};
        Matrix m2 = {{1, 2}, {-4, 2}, {0, -3}};

        EXPECT_TRUE(m1 * m2 == Matrix({{-16, 25}, {-12, 23}}));
        EXPECT_TRUE(m2 * m1 ==
                    Matrix({{24, 16, 3}, {-16, -14, -2}, {-24, -15, -3}}));
}

TEST(TEST_MATRIX, Arithmetic) {
        CheckArithmeticSum();
        CheckArithmeticDiff();
        CheckArithmeticMulti();
}

TEST(TEST_MATRIX, Transpose) {
        using Matrix = Matrix<float>;

        {
                Matrix m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                m1.Transpose();
                EXPECT_TRUE(m1 == Matrix({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}));
        }
        {
                Matrix m2 = {{0, 0}, {2, 2}, {4, 4}};

                EXPECT_TRUE(m2 == Matrix({{0, 0}, {2, 2}, {4, 4}}));
        }
}
