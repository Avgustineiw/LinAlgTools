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
class ConstSubMatrix;

template<typename T>
class SubMatrix {
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

        SubMatrix(Matrix<T>& matrix, RowSlice rows, ColumnSlice cols)
            : pmatrix_(&matrix), rows_({rows.begin, rows.end}), cols_({cols.begin, cols.end}) {
                assert(rows.begin > -1 && rows.end < matrix.Rows() &&
                       cols.begin > -1 && cols.end < matrix.Columns() &&
                       "Slice must be inside the matrix");
        }

        SubMatrix(const SubMatrix& rhs) = default;

        SubMatrix(SubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, 1})),
              cols_(std::exchange(rhs.cols_, {0, 1})) {};

        SubMatrix& operator=(const SubMatrix& lhs) = default;

        SubMatrix& operator=(SubMatrix&& rhs) noexcept {
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

        T& operator()(const Index row, const Index col) {
                assert(pmatrix_ != nullptr && "Pointer is null");
                return (*pmatrix_)(rows_.begin + row, cols_.begin + col);
        }

        ConstSubMatrix<T> ToConst() const {
                return ConstSubMatrix<T>(*pmatrix_, {rows_.begin, rows_.end},
                                         {cols_.begin, cols_.end});
        }

        friend std::ostream& operator<<(std::ostream& os, const SubMatrix& matrix) {
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
        Matrix<T>* pmatrix_;
        RowSlice rows_;
        ColumnSlice cols_;
};
}// namespace LinAlgTools
