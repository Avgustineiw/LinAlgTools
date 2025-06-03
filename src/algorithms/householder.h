#pragma once

#include "../core/math_utils.h"
#include "../core/matrix_traits.h"
#include "../types/matrix.h"
#include "../core/is_complex.h"

#include <cassert>

namespace LinAlgTools::Algorithm {
template<Core::MatrixType M>
Matrix<typename M::ElementType> HouseholderVectorReduction(const M& vector) {
        assert(vector.Rows() == 1 || vector.Columns() == 1 &&
                                             "Householder reduction is only applicable to vectors");
        using T = typename M::ElementType;

        auto result = vector;
        if constexpr (Core::IsComplexType<T>) {
                result(0, 0) -= Core::sign(vector(0, 0)) * std::complex<typename T::value_type>(vector.GetVector2Norm());
        }
        else {
                result(0, 0) -= Core::sign(vector(0, 0)) * vector.GetVector2Norm();
        }
        result.NormalizeVector();
        return {std::move(result)};
}

template<Core::MutableMatrixType F, Core::MatrixType S>
void HouseholderLeftRotation(F& matrix, const S& vector) {
        using T = typename F::ElementType;
        Matrix<T> reduced_vector = HouseholderVectorReduction(vector);
        matrix -= (T{2} * reduced_vector) * (ConjugateTransposed(reduced_vector) * matrix);
}

template<Core::MutableMatrixType F, Core::MatrixType S>
void HouseholderRightRotation(F& matrix, const S& vector) {
        using T = typename F::ElementType;
        Matrix<T> reduced_vector = HouseholderVectorReduction(vector);
        matrix -= (matrix * ConjugateTransposed(reduced_vector)) * (T{2} * reduced_vector);
}
}//namespace LinAlgTools::Algorithm

