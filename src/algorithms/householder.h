#pragma once
#include "../helpers/sign.h"
#include "../types/matrix.h"
#include "../types/submatrix.h"

#include <cmath>
#include <cstdint>
#include <utility>


//Мусорный код, который не стоит читать. Пока что скопирован из псевдокода книги Голубя, дальше буду пытаться привести к нормальному виду.
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
                                                              {0, matrix.Columns() - 1});

                vector(0, 0) -= Helpers::sgn(vector(0, 0)) * vector.Get2Norm();
                vector.Normalize();

                submatrixR -= (T{2} * vector) * (vector.Transposed() * submatrixR);
                submatrixQ -= (T{2} * vector) * (vector.Transposed() * submatrixQ);
        }

        Q.Transpose();
        return {std::move(Q), std::move(R)};
}

}// namespace LinAlgTools::Algorithm
