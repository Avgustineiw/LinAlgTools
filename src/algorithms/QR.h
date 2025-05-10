#pragma once
#include "../helpers/sign.h"
#include "../types/matrix.h"
#include "../types/submatrix.h"

#include <cmath>
#include <utility>

namespace LinAlgTools::Algorithm {
namespace Implementation {
template<typename T>
struct PairQR
{
        Matrix<T> Q;
        Matrix<T> R;
};

template<typename T>
struct GivensPair
{
        T cos = T{0};
        T sin = T{0};
};
}//namespace Implementation
using Index = Helpers::Types::Index;

template<Helpers::MatrixType F>
Implementation::PairQR<typename F::ElementType> QR_Householder(F& matrix) {
        using T = F::ElementType;

        Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> R = matrix;

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                Matrix<T> vector = R.GetSubMatrix({column, matrix.Rows() - 1},
                                                  {column, column});
                SubMatrix<T> submatrixR = R.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {column, matrix.Columns() - 1});
                SubMatrix<T> submatrixQ = Q.GetSubMatrix({column, column + vector.Rows() - 1},
                                                         {0, matrix.Rows() - 1});

                vector(0, 0) -= Helpers::sgn(vector(0, 0)) * vector.Get2Norm();
                vector.Normalize();

                submatrixR -= (T{2} * vector) * (vector.Transposed() * submatrixR);
                submatrixQ -= (T{2} * vector) * (vector.Transposed() * submatrixQ);
        }

        Q.Transpose();
        return {std::move(Q), std::move(R)};
}

template<Helpers::MatrixType F>
Implementation::PairQR<typename F::ElementType> QR_Givens(F& matrix) {
        using T = F::ElementType;

        Matrix<double> Q = Matrix<double>::Identity(matrix.Rows());
        Matrix<double> R = matrix;

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                for (Index row = matrix.Rows() - 2; row >= column; row--) {
                        T first = R(row, column);
                        T second = R(row + 1, column);
                        T cos, sin;

                        if (std::abs(second) < 1e-10) {
                                continue;
                        }

                        T norm = std::sqrt(first * first + second * second);

                        if (norm < 1e-10) {
                                cos = T{1};
                                sin = T{0};
                        }
                        else {
                                cos = first / norm;
                                sin = -second / norm;
                        }

                        Matrix<T> givens = Matrix<double>::Identity(matrix.Rows());
                        givens(row, row) = cos;
                        givens(row + 1, row + 1) = cos;
                        givens(row, row + 1) = -sin;
                        givens(row + 1, row) = sin;

                        R = givens * R;
                        Q = givens * Q;
                }
        }

        Q.Transpose();
        return {std::move(Q), std::move(R)};
}
}// namespace LinAlgTools::Algorithm
