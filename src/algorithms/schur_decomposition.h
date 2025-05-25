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
                                                             const int32_t iterations = 1000) {
        assert(!Core::IsComplexType<typename M::ElementType> &&
               "Real Schur Decomposition only for real matrices");
        assert(iterations > 0 &&
               "Number of iterations must be positive");
        assert(matrix.Rows() == matrix.Columns() &&
               "Matrix must be square for Schur's decomposition");

        auto [U, S] = HessenbergForm(matrix);
        for (int32_t k = 0; k < iterations * matrix.Columns(); k++) {
                if (Core::IsUpperTriangular(S)) break;

                auto [Q, R] = HouseholderQR(S);
                S = R * Q;
                U *= Q;
        }

        S.RemoveZeros();
        return {std::move(U), std::move(S)};
}
}//namespace LinAlgTools::Algorithm

