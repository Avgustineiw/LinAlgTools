#pragma once

#include "../types/matrix.h"
#include "../types/sub_matrix.h"
#include "givens.h"
#include "householder.h"

#include <utility>

namespace LinAlgTools::Algorithm {
namespace Implementation {
template<typename T>
struct TripletBidiagonal
{
        Matrix<T> U;
        Matrix<T> B;
        Matrix<T> V;
};
}//namespace Implementation
using Index = Core::Indices::Index;

//TODO: complex case
template<Core::MatrixType M>
Implementation::TripletBidiagonal<typename M::ElementType> Bidiagonalization(const M& matrix) {
        using T = M::ElementType;

        Matrix<T> U = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> B = matrix;
        Matrix<T> V = Matrix<T>::Identity(matrix.Columns());

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                Matrix<T> column_vector = B.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {column, column});
                SubMatrix<T> submatrixB = B.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {column, matrix.Columns() - 1});
                SubMatrix<T> submatrixU = U.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {0, matrix.Rows() - 1});
                HouseholderLeftRotation(submatrixB, column_vector);
                HouseholderLeftRotation(submatrixU, column_vector);

                Index row = column;
                if (row > matrix.Rows() - 1) break;
                Matrix<T> row_vector = B.GetSubMatrix({row, row},
                                                      {row + 1, matrix.Columns() - 1});
                submatrixB = B.GetSubMatrix({row, matrix.Rows() - 1},
                                            {row + 1, matrix.Columns() - 1});
                SubMatrix<T> submatrixV = V.GetSubMatrix({0, matrix.Columns() - 1},
                                                         {row + 1, matrix.Columns() - 1});
                HouseholderRightRotation(submatrixB, row_vector);
                HouseholderRightRotation(submatrixV, row_vector);
        }

        U.ConjugateTranspose();
        B.RemoveZeros();
        V.ConjugateTranspose();
        return {std::move(U), std::move(B), std::move(V)};
}
}// namespace LinAlgTools::Algorithm
