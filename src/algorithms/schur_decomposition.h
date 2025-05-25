#pragma once

#include "../core/matrix_traits.h"
#include "../types/matrix.h"
#include "hessenberg.h"
#include "qr_decomposition.h"

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
                                                             const size_t iterations = 10) {
        assert(!Core::IsComplexType<typename M::ElementType> &&
               "Real Schur Decomposition only for real matrices");
        assert(iterations > 0 &&
               "Number of iterations must be positive");
        assert(matrix.Rows() == matrix.Columns() &&
               "Matrix must be square for Schur's decomposition");

        auto [U, S] = HessenbergForm(matrix);
        for (size_t k = 0; k < iterations * matrix.Rows() * matrix.Columns(); k++) {
                if (Core::IsUpperTriangular(S)) break;

                auto [Q, R] = GivensQR(S);
                S = R * Q;
                U = U * Q;
        }

        S.RemoveZeros();
        return {std::move(U), std::move(S)};
}
}//namespace LinAlgTools::Algorithm

