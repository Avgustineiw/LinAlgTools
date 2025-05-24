#pragma once

#include "../types/matrix.h"
#include "../core/matrix_traits.h"
#include "qr_decomposition.h"
#include  "bidiagonalization.h"

namespace LinAlgTools::Algorithm {
namespace Implementation {
template<typename T>
struct TripletSVD
{
        Matrix<T> U;
        Matrix<T> S;
        Matrix<T> VT;
};
}//namespace Implementation

template<Core::MatrixType M>
Implementation::TripletSVD<typename M::ElementType> NaiveSVD(const M& matrix,
                                                             const size_t iterations = 100) {
        assert(iterations > 0 &&
               "Number of iterations must be positive");
        using T = M::ElementType;

        auto [U, S1, V] = Bidiagonalization(matrix);
        V.ConjugateTranspose();
        for (size_t k = 0; k < iterations; k++) {
                auto [Q1, R1] = HouseholderQR(S1);
                auto [Q2, R2] = HouseholderQR(R1.ConjugateTransposed());
                S1 = R2.ConjugateTransposed();
                U *= Q1;
                V *= Q2;
        }
        V.ConjugateTranspose();
        S1.RemoveZeros();
        return {std::move(U), std::move(S1), std::move(V)};
}
}//namespace LinAlgTools::Algorithm

