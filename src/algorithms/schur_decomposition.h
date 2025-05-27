#pragma once

#include "../core/matrix_traits.h"
#include "../types/matrix.h"
#include "hessenberg_form.h"
#include "qr_decomposition.h"
#include "wilkinson_shift.h"

namespace LinAlgTools::Algorithm {
namespace Implementation {
template<typename T>
struct PairSchur
{
        Matrix<T> U;
        Matrix<T> S;
};
}//namespace Implementation

template<Core::MatrixType M>
Implementation::PairSchur<typename M::ElementType> RealSchur(const M& matrix,
                                                             const int32_t iterations = 50) {
        assert(!Core::IsComplexType<typename M::ElementType> &&
               "Real Schur Decomposition only for real matrices");
        assert(iterations > 0 &&
               "Number of iterations must be positive");
        assert(matrix.Rows() == matrix.Columns() &&
               "Matrix must be square for Schur's decomposition");
        using T = typename M::ElementType;

        auto [U, S] = HessenbergForm(matrix);
        int32_t iteration = 0;

        do {
                T shift = GetWilkinsonShift(S);
                auto [Q, R] = HouseholderQR(S - shift * Matrix<T>::Identity(matrix.Rows()));
                S = R * Q + shift * Matrix<T>::Identity(matrix.Rows());
                U *= Q;
                iteration++;
        } while (!Core::IsUpperTriangular(S) &&
                 iteration < iterations * matrix.Columns());

        S.RemoveZeros();
        return {std::move(U), std::move(S)};
}
}//namespace LinAlgTools::Algorithm

