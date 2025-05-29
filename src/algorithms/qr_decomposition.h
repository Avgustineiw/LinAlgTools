#pragma once

#include "../core/math_utils.h"
#include "../types/matrix.h"
#include "../types/sub_matrix.h"
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
using Index = Core::Indices::Index;

template<Core::MatrixType M>
Implementation::PairQR<typename M::ElementType> HessenbergQR(const M& matrix) {
        assert(Core::IsUpperHessenberg(matrix) &&
               "Hessenberg QR for hessenberg matrix.");
        using T = typename M::ElementType;

        Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> R = matrix;

        for (Index row = 0; row < std::min(R.Rows() - 1, R.Columns()); row++) {
                auto first = R(row, row);
                auto second = R(row + 1, row);

                SubMatrix<T> submatrixR = R.GetSubMatrix({0, matrix.Rows() - 1},
                                                         {row, matrix.Columns() - 1});
                SubMatrix<T> submatrixQ = Q.GetSubMatrix({0, matrix.Rows() - 1},
                                                         {0, matrix.Rows() - 1});

                GivensLeftRotation(submatrixR, row, row + 1, first, second);
                GivensRightRotation(submatrixQ, row, row + 1, first, second);
        }

        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}

template<Core::MatrixType M>
Implementation::PairQR<typename M::ElementType> HouseholderQR(const M& matrix) {
        using T = typename M::ElementType;

        if (Core::IsUpperTriangular(matrix)) {
                return {std::move(Matrix<T>::Identity(matrix.Rows())), std::move(matrix)};
        }
        if (Core::IsUpperHessenberg(matrix)) {
                return HessenbergQR(matrix);
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

                HouseholderLeftRotation(submatrixR, vector);
                HouseholderLeftRotation(submatrixQ, vector);
        }

        Q.ConjugateTranspose();
        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}

template<Core::MatrixType M>
Implementation::PairQR<typename M::ElementType> GivensQR(const M& matrix) {
        using T = typename M::ElementType;

        if (Core::IsUpperTriangular(matrix)) {
                return {std::move(Matrix<T>::Identity(matrix.Rows())), std::move(matrix)};
        }
        if (Core::IsUpperHessenberg(matrix)) {
                return HessenbergQR(matrix);
        }

        Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> R = matrix;

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                for (Index row = matrix.Rows() - 2; row >= column; row--) {
                        T first = R(row, column);
                        T second = R(row + 1, column);

                        if (Core::IsZero(second)) continue;

                        SubMatrix<T> submatrixR = R.GetSubMatrix({0, matrix.Rows() - 1},
                                                                 {column, matrix.Columns() - 1});
                        SubMatrix<T> submatrixQ = Q.GetSubMatrix({0, matrix.Rows() - 1},
                                                                 {0, matrix.Rows() - 1});

                        GivensLeftRotation(submatrixR, row, row + 1, first, second);
                        GivensRightRotation(submatrixQ, row, row + 1, first, second);
                }
        }

        R.RemoveZeros();
        return {std::move(Q), std::move(R)};
}
}// namespace LinAlgTools::Algorithm

