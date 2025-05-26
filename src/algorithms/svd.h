#pragma once

#include "../core/matrix_traits.h"
#include "../core/indices.h"
#include "../core/matrix_utils.h"
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
                        Core::SwapColumns(U, j, j + 1);
                        Core::SwapRows(VT, j, j + 1);
                }
        }
}
}//namespace Implementation

template<Core::MatrixType M>
Implementation::TripletSVD<typename M::ElementType> NaiveSVD(const M& matrix,
                                                             const int32_t iterations = 50) {
        assert(iterations > 0 &&
               "Number of iterations must be positive");

        auto [U, S, VT] = Bidiagonalization(matrix);
        for (int32_t k = 0; k < iterations * matrix.Columns(); k++) {
                auto [Q1, R1] = HouseholderQR(S);
                auto [Q2, R2] = HouseholderQR(R1.ConjugateTransposed());
                S = R2.ConjugateTransposed();
                U *= Q1;
                VT *= Q2;
        }

        VT.ConjugateTranspose();
        Implementation::SortSingularValues(U, S, VT);
        S.RemoveZeros();
        return {std::move(U), std::move(S), std::move(VT)};
}
}//namespace LinAlgTools::Algorithm

