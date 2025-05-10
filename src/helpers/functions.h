#include "matrix_type.h"

#include "types.h"
#include <cstdlib>

namespace LinAlgTools::Helpers {
namespace Functions {
using Index = Types::Index;

template<MatrixType F, MatrixType S>
bool AreEqualMatrices(const F& lhs, const S& rhs) {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns()) {
                return false;
        }

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        if (std::abs(lhs(i, j) - rhs(i, j)) > 1e-5) {
                                return false;
                        }
                }
        }

        return true;
}

template<MatrixType M>
bool IsOrthogonal(const M& matrix) {
        for (Index row = 1; row <= matrix.Columns(); row++) {
                if (std::abs(matrix.GetColumn(row).Get2Norm() - 1) > 1e-5) {
                        return false;
                }
        }

        return true;
}

template<MatrixType M>
bool IsUpperTriangular(const M& matrix) {
        for (Index i = 0; i < matrix.Rows(); ++i) {
                for (Index j = 0; j < i && j < matrix.Columns(); ++j) {
                        if (std::abs(matrix(i, j)) > 1e-5) {
                                return false;
                        }
                }
        }
        return true;
}
}//namespace Functions
}//namespace LinAlgTools::Helpers
