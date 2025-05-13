#pragma once

#include "../helpers/functions.h"
#include "../types/matrix.h"
#include "../types/submatrix.h"
#include "givens.h"
#include "householder.h"

#include <utility>

namespace LinAlgTools::Algorithm {
namespace Implementation {
template<typename T>
struct PairQR
{
        Matrix<T> Q;
        Matrix<T> R;
};
}//namespace Implementation
using Index = Helpers::Types::Index;

template<Helpers::MatrixType F>
Implementation::PairQR<typename F::ElementType> HouseholderQR(const F& matrix) {
        using T = F::ElementType;

        if (Helpers::Functions::IsUpperTriangular(matrix)) {
                return {std::move(Matrix<T>::Identity(matrix.Rows())), std::move(matrix)};
        }


        Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> R = matrix;

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                Matrix<T> vector = R.GetSubMatrix({column, matrix.Rows() - 1},
                                                  {column, column});
                SubMatrix<T> submatrixR = R.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {column, matrix.Columns() - 1});
                SubMatrix<T> submatrixQ = Q.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {0, matrix.Rows() - 1});

                HouseholderVectorReduction(vector);
                HouseholderLeftRotation(submatrixR, vector);
                HouseholderLeftRotation(submatrixQ, vector);
        }

        Q.Transpose();
        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}

template<Helpers::MatrixType M>
Implementation::PairQR<typename M::ElementType> GivensQR(const M& matrix) {
        using T = typename M::ElementType;

        if (Helpers::Functions::IsUpperTriangular(matrix)) {
                return {std::move(Matrix<T>::Identity(matrix.Rows())), std::move(matrix)};
        }

        Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> R = matrix;

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                for (Index row = matrix.Rows() - 2; row >= column; row--) {
                        T first = R(row, column);
                        T second = R(row + 1, column);

                        if (Helpers::IsZero(second)) continue;

                        GivensLeftRotation(R, row, column, first, second);
                        GivensRightRotation(Q, row, 0, first, second);
                }
        }

        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}
}// namespace LinAlgTools::Algorithm
