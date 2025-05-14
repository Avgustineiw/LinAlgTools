#pragma once

#include "../core/math_utils.h"
#include "../core/matrix_traits.h"
#include "../types/matrix.h"

#include <cassert>

namespace LinAlgTools::Algorithm {
template<Core::MatrixType M>
Matrix<typename M::ElementType> HouseholderVectorReduction(const M& vector) {
        assert(vector.Rows() == 1 || vector.Columns() == 1 &&
                                             "Householder reduction is only applicable to vectors");

        auto result = vector;
        result(0, 0) -= Core::sign(vector(0, 0)) * vector.Get2Norm();
        result.Normalize();
        return result;
}

template<Core::MutableMatrixType F, Core::MatrixType S>
void HouseholderLeftRotation(F& matrix, const S& vector) {
        using T = F::ElementType;
        auto reduced_vector = HouseholderVectorReduction(vector);
        matrix -= (T{2} * reduced_vector) * (reduced_vector.ConjugateTransposed() * matrix);
}

template<Core::MutableMatrixType F, Core::MatrixType S>
void HouseholderRightRotation(F& matrix, const S& vector) {
        using T = F::ElementType;
        auto reduced_vector = HouseholderVectorReduction(vector);
        matrix -= (matrix * reduced_vector.ConjugateTransposed()) * (T{2} * reduced_vector);
}
}//namespace LinAlgTools::Algorithm
