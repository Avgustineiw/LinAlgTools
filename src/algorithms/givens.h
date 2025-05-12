#pragma once

#include "../helpers/functions.h"
#include "../types/matrix.h"

#include <cmath>

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
        T norm = std::hypot(first, second);
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
                matrix(row, i) = cos * upper - sin * lower;
                matrix(row + 1, i) = sin * upper + cos * lower;
        }
}

template<Helpers::MatrixType M>
void GivensRightRotation(M& matrix,
                         Index row, Index column,
                         typename M::ElementType first, typename M::ElementType second) {
        using T = M::ElementType;

        auto [cos, sin] = CalculateGivensPair(first, second);
        for (Index i = 0; i < matrix.Rows(); i++) {
                T left = matrix(i, row);
                T right = matrix(i, row + 1);
                matrix(i, row) = cos * left - sin * right;
                matrix(i, row + 1) = sin * left + cos * right;
        }
}
}//namespace LinAlgTools::Algorithm
