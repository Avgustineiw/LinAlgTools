#pragma once

#include "../helpers/functions.h"
#include "../helpers/matrix_type.h"
#include "../types/matrix.h"

#include <cassert>

namespace LinAlgTools {
template<Helpers::MatrixType M>
Matrix<typename M::ElementType> HouseholderVectorReduction(const M& vector) {
        assert(vector.Rows() == 1 || vector.Columns() == 1 &&
                                             "Householder reduction is only applicable to vectors");

        auto result = vector;
        result(0, 0) -= Helpers::sign(vector(0, 0)) * vector.Get2Norm();
        result.Normalize();
        return result;
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
void HouseholderLeftRotation(F& matrix, const S& vector) {
        using T = F::ElementType;
        auto reduced_vector = HouseholderVectorReduction(vector);
        matrix -= (T{2} * reduced_vector) * (reduced_vector.ConjugateTransposed() * matrix);
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
void HouseholderRightRotation(F& matrix, const S& vector) {
        using T = F::ElementType;
        auto reduced_vector = HouseholderVectorReduction(vector);
        matrix -= (matrix * reduced_vector.ConjugateTransposed()) * (T{2} * reduced_vector);
}
}//namespace LinAlgTools
