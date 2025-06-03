#pragma once

#include "../core/indices.h"
#include "../core/matrix_traits.h"
#include "../types/matrix.h"
#include "bidiagonalization.h"
#include "qr_decomposition.h"

#include <complex>

namespace LinAlgTools::Algorithm {
namespace Implementation {
using Index = Core::Indices::Index;

template<typename T>
struct TripletSVD
{
        Matrix<T> U;
        Matrix<T> S;
        Matrix<T> VT;
};

template<Core::MutableMatrixType M>
void SortSingularValues(M& U, M& S, M& VT) {
        using T = typename M::ElementType;
        for (Index i = 0; i < std::min(S.Rows(), S.Columns()); i++) {
                for (Index j = 0; j < std::min(S.Rows(), S.Columns()) - i - 1; j++) {
                        if (std::real(S(j, j)) >= std::real(S(j + 1, j + 1))) continue;

                        std::swap(S(j, j), S(j + 1, j + 1));
                        SwapColumns(U, j, j + 1);
                        SwapRows(VT, j, j + 1);
                }
        }
}
}//namespace Implementation

template<Core::MatrixType M>
Implementation::TripletSVD<typename M::ElementType> NaiveSVD(const M& matrix,
                                                             const int32_t iterations = 10) {
        assert(iterations > 0 &&
               "Number of iterations must be positive");

        auto [U, S, VT] = Bidiagonalization(matrix);
        VT.ConjugateTranspose();
        int32_t iteration = 0;

        do {
                auto [Q1, R1] = HouseholderQR(S);
                auto [Q2, R2] = HouseholderQR(ConjugateTransposed(R1));
                S = ConjugateTransposed(R2);
                U *= Q1;
                VT *= Q2;
                iteration++;
        } while (!Core::IsDiagonal(S) &&
                 iteration < iterations * matrix.Columns());

        VT.ConjugateTranspose();
        Implementation::SortSingularValues(U, S, VT);
        S.RemoveZeros();
        return {std::move(U), std::move(S), std::move(VT)};
}
}//namespace LinAlgTools::Algorithm

