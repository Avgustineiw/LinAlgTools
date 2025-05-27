#pragma once

#include "../core/indices.h"
#include "../core/matrix_traits.h"
#include "../types/matrix.h"
#include <iostream>

namespace LinAlgTools::Algorithm {
template<Core::MatrixType M>
typename M::ElementType GetWilkinsonShift(const M& matrix) {
        using T = typename M::ElementType;
        using Index = Core::Indices::Index;

        Index n = matrix.Rows();
        Index m = n - 1;
        T a_mm = matrix(m, m);
        T a_mn = matrix(m, n);
        T a_nm = matrix(n, m);
        T a_nn = matrix(n, n);

        T delta = (a_mm - a_nn) / T{2};
        T numerator = a_nn - Core::sign(delta) * a_mn * a_mn;
        T denominator = std::abs(delta) + std::sqrt(delta * delta + a_mn * a_nm);
        T result = numerator / denominator;
        return Core::IsZero(denominator) ? 0 : result;
}
}//namespace LinAlgTools::Algorithm
