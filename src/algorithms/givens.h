#pragma once

#include "../helpers/functions.h"
#include "../types/matrix.h"

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

template<typename T>
Implementation::GivensPair<T> CalculateGivensPair(T first, T second) {
        T norm = std::sqrt(std::norm(first) + std::norm(second));
        if (Helpers::IsZero(norm)) {
                return {T{1}, T{0}};
        }
        return {first / norm, -second / norm};
}

template<Helpers::MatrixType M>
void GivensLeftRotation(M& matrix,
                        Index row, Index column,
                        typename M::ElementType first, typename M::ElementType second) {
        using T = M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = column; i < matrix.Columns(); i++) {
                T upper = matrix(row, i);
                T lower = matrix(row + 1, i);
                if constexpr (Helpers::IsComplexType<T>) {
                        matrix(row, i) = std::conj(cos) * upper - std::conj(sin) * lower;
                }
                else {
                        matrix(row, i) = cos * upper - sin * lower;
                }
                matrix(row + 1, i) = sin * upper + cos * lower;
        }
}

template<Helpers::MatrixType M>
void GivensRightRotation(M& matrix,
                         Index column, Index row,
                         typename M::ElementType first, typename M::ElementType second) {
        using T = typename M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = row; i < matrix.Rows(); i++) {
                T left = matrix(i, column);
                T right = matrix(i, column + 1);
                matrix(i, column) = cos * left - sin * right;
                if constexpr (Helpers::IsComplexType<T>) {
                        matrix(i, column + 1) = std::conj(sin) * left + std::conj(cos) * right;
                }
                matrix(i, column + 1) = sin * left + cos * right;
        }
}
}//namespace LinAlgTools::Algorithm
