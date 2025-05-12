#pragma once
#include "../helpers/functions.h"
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
        T cos = T{1};
        T sin = T{0};
};
}//namespace Implementation
using Index = Helpers::Types::Index;

template<Helpers::MatrixType F>
Implementation::PairQR<typename F::ElementType> QR_Householder(const F& matrix) {
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
        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}

template<typename T>
Implementation::GivensPair<T> CalculateGivensPair(T first, T second) {
        T norm = std::sqrt(first * first + second * second);
        if (Helpers::IsZero(norm)) {
                return {T{1}, T{0}};
        }
        return {first / norm, -second / norm};
}

template<Helpers::MatrixType M>
void GivensLeftRotation(M& matrix,
                        Index row, Index column,
                        typename M::ElementType first,
                        typename M::ElementType second) {
        using T = M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = column; i < matrix.Columns(); i++) {
                T upper = matrix(row, i);
                T lower = matrix(row + 1, i);
                matrix(row, i) = cos * upper - sin * lower;
                matrix(row + 1, i) = sin * upper + cos * lower;
        }
}

template<Helpers::MatrixType M>
void GivensRightRotation(M& matrix,
                         Index row, Index column,
                         typename M::ElementType first,
                         typename M::ElementType second) {
        using T = M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = 0; i < matrix.Rows(); i++) {
                T left = matrix(i, row);
                T right = matrix(i, row + 1);
                matrix(i, row) = cos * left - sin * right;
                matrix(i, row + 1) = sin * left + cos * right;
        }
}


template<Helpers::MatrixType M>
Implementation::PairQR<typename M::ElementType> QR_Givens(const M& matrix) {
        using T = typename M::ElementType;

        Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> R = matrix;

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                for (Index row = matrix.Rows() - 2; row >= column; row--) {
                        T first = R(row, column);
                        T second = R(row + 1, column);

                        if (Helpers::IsZero(second)) continue;

                        GivensLeftRotation(R, row, column, first, second);
                        GivensRightRotation(Q, row, column, first, second);
                }
        }

        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}
}// namespace LinAlgTools::Algorithm
