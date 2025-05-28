#pragma once

#include "../core/indices.h"
#include "../core/matrix_traits.h"
#include "../core/math_utils.h"

namespace LinAlgTools::Algorithm {
template<Core::MatrixType M>
typename M::ElementType GetWilkinsonShift(const M& matrix) {
        using T = typename M::ElementType;
        using Index = Core::Indices::Index;

        Index n = matrix.Rows();
        Index m = n - 1;
        T a_mm = matrix(m, m);
        T a_mn = matrix(m, n);
        T a_nn = matrix(n, n);

        T delta = (a_mm - a_nn) / T{2};
        T numerator = Core::sign(delta) * a_mn * a_mn;
        T denominator = std::abs(delta) + std::sqrt(delta * delta + a_mn * a_mn);
        T result = a_nn - numerator / denominator;
        return (!Core::IsZero(denominator)) ? result : a_nn;
}
}//namespace LinAlgTools::Algorithm
