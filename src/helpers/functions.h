#pragma once

#include "matrix_type.h"

#include "types.h"
#include <cstdlib>

namespace LinAlgTools::Helpers {
constexpr double EPSILON = 1e-10;

template<typename T>
bool IsZero(T value) {
        return std::abs(value) < EPSILON;
}

template<typename T>
int sign(T val) {
        return (T(0) < val) - (val < T(0));
}

namespace Functions {
using Index = Types::Index;

template<MatrixType F, MatrixType S>
bool AreEqualMatrices(const F& lhs, const S& rhs) {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns()) {
                return false;
        }

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        if (!IsZero(lhs(i, j) - rhs(i, j))) {
                                return false;
                        }
                }
        }

        return true;
}

template<MatrixType M>
bool IsOrthogonal(const M& matrix) {
        for (Index row = 1; row <= matrix.Columns(); row++) {
                if (!IsZero(matrix.GetColumn(row).Get2Norm() - 1)) {
                        return false;
                }
        }

        return true;
}

template<MatrixType M>
bool IsUpperTriangular(const M& matrix) {
        for (Index i = 0; i < matrix.Rows(); ++i) {
                for (Index j = 0; j < i && j < matrix.Columns(); ++j) {
                        if (!IsZero(matrix(i, j))) {
                                return false;
                        }
                }
        }
        return true;
}
}//namespace Functions
}//namespace LinAlgTools::Helpers
