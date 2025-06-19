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
struct PairHessenberg
{
        Matrix<T> U;
        Matrix<T> H;
};
}//namespace Implementation
using Index = Core::Indices::Index;

template<Core::MatrixType M>
Implementation::PairHessenberg<typename M::ElementType> HessenbergForm(const M& matrix) {
        assert(matrix.Rows() == matrix.Columns() &&
               "Hessenberg form cannot be computer for non-square matrices");
        using T = typename M::ElementType;
        if (Core::IsUpperHessenberg(matrix)) {
                return {std::move(Matrix<T>::Identity(matrix.Rows())), std::move(matrix)};
        }

        Matrix<T> U = Matrix<T>::Identity(matrix.Rows());
        Matrix<T> H = matrix;

        for (Index column = 0; column < matrix.Rows() - 1; column++) {
                Matrix<T> vector = H.GetSubMatrix({column + 1, matrix.Rows() - 1},
                                                  {column, column});
                SubMatrix<T> submatrixH_left = H.GetSubMatrix({column + 1, matrix.Rows() - 1},
                                                              {column, matrix.Columns() - 1});
                SubMatrix<T> submatrixH_right = H.GetSubMatrix({0, matrix.Rows() - 1},
                                                               {column + 1, matrix.Columns() - 1});
                SubMatrix<T> submatrixU = U.GetSubMatrix({column + 1, matrix.Rows() - 1},
                                                         {0, matrix.Rows() - 1});

                HouseholderLeftRotation(submatrixH_left, vector);
                HouseholderLeftRotation(submatrixU, vector);
                HouseholderRightRotation(submatrixH_right, vector.ConjugateTranspose());
        }

        U.ConjugateTranspose();
        H.RemoveZeros();
        return {std::move(U), std::move(H)};
}
}// namespace LinAlgTools::Algorithm

