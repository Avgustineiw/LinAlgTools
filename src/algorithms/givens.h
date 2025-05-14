#pragma once

#include "../core/indices.h"
#include "../core/math_utils.h"

#include <cmath>
#include <complex>

namespace LinAlgTools::Algorithm {
namespace Implementation {
template<typename T>
struct GivensPair
{
        T cos = T{1};
        T sin = T{0};
};
}//namespace Implementation

using Index = Core::Indices::Index;

template<typename T>
Implementation::GivensPair<T> CalculateGivensPair(T first, T second) {
        T norm = std::sqrt(std::norm(first) + std::norm(second));
        if (Core::IsZero(norm)) {
                return {T{1}, T{0}};
        }
        return {first / norm, -second / norm};
}

template<Core::MutableMatrixType M>
void GivensLeftRotation(M& matrix,
                        Index upper_row, Index lower_row,
                        typename M::ElementType first, typename M::ElementType second) {
        using T = M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = 0; i < matrix.Columns(); i++) {
                T upper = matrix(upper_row, i);
                T lower = matrix(lower_row, i);

                if constexpr (Core::IsComplexType<T>) {
                        matrix(upper_row, i) = std::conj(cos) * upper - std::conj(sin) * lower;
                }
                else {
                        matrix(upper_row, i) = cos * upper - sin * lower;
                }
                matrix(lower_row, i) = sin * upper + cos * lower;
        }
}

template<Core::MutableMatrixType M>
void GivensRightRotation(M& matrix,
                         Index left_column, Index right_column,
                         typename M::ElementType first, typename M::ElementType second) {
        using T = typename M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = 0; i < matrix.Rows(); i++) {
                T left = matrix(i, left_column);
                T right = matrix(i, right_column);

                matrix(i, left_column) = cos * left - sin * right;
                if constexpr (Core::IsComplexType<T>) {
                        matrix(i, right_column) = std::conj(sin) * left + std::conj(cos) * right;
                }
                else {
                        matrix(i, right_column) = sin * left + cos * right;
                }
        }
}
}//namespace LinAlgTools::Algorithm
