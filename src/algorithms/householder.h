#pragma once

#include "../helpers/functions.h"
#include "../helpers/matrix_type.h"

#include <cassert>

namespace LinAlgTools {
template<Helpers::MutableMatrixType M>
void HouseholderVectorReduction(M& vector) {
        assert(vector.Rows() == 1 || vector.Columns() == 1 &&
                                            "Householder reduction is only applicable to vectors");

        vector(0, 0) -= Helpers::sign(vector(0, 0)) * vector.Get2Norm();
        vector.Normalize();
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
void HouseholderLeftRotation(F& matrix, const S& vector) {
        using T = F::ElementType;
        matrix -= (T{2} * vector) * (vector.ConjugateTransposed() * matrix);
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
void HouseholderRightRotation(F& matrix, const S& vector) {
        using T = F::ElementType;
        matrix -= (matrix * vector.ConjugateTransposed()) * (T{2} * vector);
}
}//namespace LinAlgTools
