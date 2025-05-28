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
        Matrix<T> VT;
};
}//namespace Implementation
using Index = Core::Indices::Index;

template<Core::MatrixType M>
Implementation::TripletBidiagonal<typename M::ElementType> Bidiagonalization(const M& matrix) {
        using T = typename M::ElementType;

        Matrix<T> U = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> B = matrix;
        Matrix<T> VT = Matrix<T>::Identity(matrix.Columns());

        for (Index column = 0; column < std::min(matrix.Rows(), matrix.Columns()); column++) {
                Matrix<T> column_vector = B.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {column, column});
                SubMatrix<T> submatrixB = B.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {column, matrix.Columns() - 1});
                SubMatrix<T> submatrixU = U.GetSubMatrix({column, matrix.Rows() - 1},
                                                         {0, matrix.Rows() - 1});
                HouseholderLeftRotation(submatrixB, column_vector);
                HouseholderLeftRotation(submatrixU, column_vector);

                if (column > matrix.Columns() - 2) continue;
                Matrix<T> row_vector = B.GetSubMatrix({column, column},
                                                      {column + 1, matrix.Columns() - 1});
                submatrixB = B.GetSubMatrix({column, matrix.Rows() - 1},
                                            {column+ 1, matrix.Columns() - 1});
                SubMatrix<T> submatrixVT = VT.GetSubMatrix({0, matrix.Columns() - 1},
                                                           {column+ 1, matrix.Columns() - 1});
                HouseholderRightRotation(submatrixB, row_vector);
                HouseholderRightRotation(submatrixVT, row_vector);
        }

        U.ConjugateTranspose();
        B.RemoveZeros();
        VT.ConjugateTranspose();
        return {std::move(U), std::move(B), std::move(VT)};
}
}// namespace LinAlgTools::Algorithm

