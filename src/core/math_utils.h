#pragma once

#include "indices.h"
#include "is_complex.h"
#include "matrix_traits.h"

#include <cmath>
#include <complex>
#include <cstdlib>

namespace LinAlgTools::Core {
using Index = Indices::Index;

inline constexpr long double EPSILON = 1e-10;

template<typename T>
bool IsZero(const T& value) {
        if constexpr (IsComplexType<T>) {
                return std::norm(value) < EPSILON;
        }
        else {
                return std::abs(value) < EPSILON;
        }
}

template<typename T>
T sign(T value) {
        if (IsZero(value)) {
                return T{1};
        }

        if constexpr (IsComplexType<T>) {
                return value / std::sqrt(std::norm(value));
        }
        else {
                return value > T{0} ? T{1} : T{-1};
        }
}

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
        using T = typename M::ElementType;
        for (Index row = 1; row <= matrix.Columns(); row++) {
                if (!IsZero(matrix.GetColumn(row).GetVector2Norm() - T{1})) {
                        return false;
                }
        }

        return true;
}

template<MatrixType M>
bool IsUnitary(const M& matrix) {
        return AreEqualMatrices(matrix * matrix.ConjugateTransposed(),
                                Matrix<typename M::ElementType>::Identity(matrix.Rows()));
}

template<MatrixType M>
bool IsDiagonal(const M& matrix) {
        for (Index i = 0; i < matrix.Rows(); i++) {
                for (Index j = 0; j < matrix.Columns(); j++) {
                        if (i == j) continue;

                        if (!IsZero(matrix(i, j))) {
                                return false;
                        }
                }
        }

        return true;
}

template<MatrixType M>
bool IsUpperTriangular(const M& matrix) {
        for (Index i = 1; i < matrix.Rows(); ++i) {
                for (Index j = 0; j < i && j < matrix.Columns(); ++j) {
                        if (!IsZero(matrix(i, j))) {
                                return false;
                        }
                }
        }
        return true;
}

template<MatrixType M>
bool IsUpperHessenberg(const M& matrix) {
        for (Index i = 2; i < matrix.Rows(); ++i) {
                for (Index j = 0; j < i - 1; ++j) {
                        if (!IsZero(matrix(i, j))) {
                                return false;
                        }
                }
        }
        return true;
}
}// namespace LinAlgTools::Core

