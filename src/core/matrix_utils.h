#pragma once

#include "matrix_traits.h"
#include "indices.h"
#include <cassert>
#include <utility>

namespace LinAlgTools::Core{
using Index = Indices::Index;

template<Core::MutableMatrixType M>
void SwapColumns(M& matrix, Index first, Index second) {
        assert(first >= 0 && first < matrix.Columns() &&
               second >= 0 && second < matrix.Columns() &&
               "Invalid column index");

        for (Index i = 0; i < matrix.Rows(); i++) {
                std::swap(matrix(i, first), matrix(i, second));
        }
}

template<Core::MutableMatrixType M>
void SwapRows(M& matrix, Index first, Index second) {
        assert(first >= 0 && first < matrix.Rows() &&
               second >= 0 && second < matrix.Rows() &&
               "Invalid column index");

        for (Index i = 0; i < matrix.Columns(); i++) {
                std::swap(matrix(first, i), matrix(second, i));
        }
}
} //namespace LinAlgTools::Core

