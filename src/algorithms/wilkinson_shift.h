#pragma once

#include "../core/indices.h"
#include "../core/matrix_traits.h"
#include "../types/matrix.h"

namespace LinAlgTools::Algorithm {
template<Core::MatrixType M>
typename M::ElementType GetWilkinsonShift(const M& matrix) {
        using T = typename M::ElementType;
        using Index = Core::Index;

        Index n = matrix.Rows();
        Index m = n - 1;
        T a_mm = matrix(m - 1, m - 1);
        T a_mn = matrix(m - 1, m);
        T a_nm = matrix(m, m - 1);
        T a_nn = matrix(m, m);

        T delta = (a_mm - a_nn) / 2.0;
        return a_nn - Core::sign(delta) * a_mn * a_mn / (std::abs(delta) + std::sqrt(delta * delta + a_mn * a_nm));
}
}//namespace LinAlgTools::Algorithm
