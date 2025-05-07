#pragma once

#include "../helpers/index.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace LinAlgTools {
template<typename T>
class Matrix;
template<typename T>
class SubMatrix;

template<typename T>
class ConstSubMatrix {
        using Index = Helpers::Types::Index;

        struct RowSlice
        {
                Index begin = -1;
                Index end = -1;
        };
        struct ColumnSlice
        {
                Index begin = -1;
                Index end = -1;
        };

public:
        using ElementType = std::remove_cv_t<T>;

        ConstSubMatrix(const Matrix<T>& matrix, RowSlice rows, ColumnSlice cols)
            : pmatrix_(&matrix), rows_({rows.begin, rows.end}), cols_({cols.begin, cols.end}) {
                assert(rows.begin > -1 && rows.end < matrix.Rows() &&
                       cols.begin > -1 && cols.end < matrix.Columns() &&
                       "Slice must be inside the matrix");
        }

        ConstSubMatrix(const ConstSubMatrix& rhs) = default;

        ConstSubMatrix(ConstSubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, 1})),
              cols_(std::exchange(rhs.cols_, {0, 1})) {};

        ConstSubMatrix& operator=(const ConstSubMatrix& lhs) = default;

        ConstSubMatrix& operator=(ConstSubMatrix&& rhs) noexcept {
                pmatrix_ = std::exchange(rhs.pmatrix_, nullptr);
                rows_ = std::exchange(rhs.rows_, {0, 1});
                cols_ = std::exchange(rhs.cols_, {0, 1});
                return *this;
        }

        Index Rows() const {
                return rows_.end - rows_.begin + 1;
        }

        Index Columns() const {
                return cols_.end - cols_.begin + 1;
        }

        T operator()(Index row, Index col) const {
                assert(pmatrix_ != nullptr && "Pointer is null");
                return (*pmatrix_)(rows_.begin + row, cols_.begin + col);
        }


        friend std::ostream& operator<<(std::ostream& os, const ConstSubMatrix& matrix) {
                for (Index i = 0; i < matrix.Rows(); i++) {
                        os << '[';
                        for (Index j = 0; j < matrix.Columns(); j++) {
                                os << matrix(i, j);
                                if (j + 1 < matrix.Columns()) {
                                        os << ", ";
                                }
                        }
                        os << ']';
                        if (i + 1 < matrix.Rows()) {
                                os << '\n';
                        }
                }

                return os;
        }


private:
        const Matrix<T>* pmatrix_;
        RowSlice rows_;
        ColumnSlice cols_;
};
}//namespace LinAlgTools
