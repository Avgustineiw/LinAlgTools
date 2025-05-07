#pragma once

#include "../helpers/matrix_type.h"
#include "../helpers/types.h"

#include <cassert>
#include <cstddef>
#include <ostream>

namespace LinAlgTools {
template<typename T>
class SubMatrix {
        using Index = Helpers::Types::Index;
        using RowSlice = Helpers::Types::RowSlice;
        using ColumnSlice = Helpers::Types::ColumnSlice;

public:
        using ElementType = std::remove_cv_t<T>;

        SubMatrix(Matrix<T>& matrix, RowSlice rows, ColumnSlice columns)
            : pmatrix_(&matrix),
              rows_({rows.begin, rows.end}),
              columns_({columns.begin, columns.end}) {
                assert(rows.begin > -1 && rows.end < matrix.Rows() &&
                       columns.begin > -1 && columns.end < matrix.Columns() &&
                       "Slice must be inside the matrix");
        }

        SubMatrix(const SubMatrix& rhs) = default;

        SubMatrix(SubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, 1})),
              columns_(std::exchange(rhs.columns_, {0, 1})) {};

        SubMatrix& operator=(const SubMatrix& lhs) = default;

        SubMatrix& operator=(SubMatrix&& rhs) noexcept {
                pmatrix_ = std::exchange(rhs.pmatrix_, nullptr);
                rows_ = std::exchange(rhs.rows_, {0, 1});
                columns_ = std::exchange(rhs.columns_, {0, 1});
                return *this;
        }

        Index Rows() const {
                return rows_.end - rows_.begin + 1;
        }

        Index Columns() const {
                return columns_.end - columns_.begin + 1;
        }

        T operator()(Index row, Index col) const {
                assert(pmatrix_ != nullptr &&
                       "Pointer is null");
                return (*pmatrix_)(rows_.begin + row, columns_.begin + col);
        }

        T& operator()(const Index row, const Index col) {
                assert(pmatrix_ != nullptr &&
                       "Pointer is null");
                return (*pmatrix_)(rows_.begin + row, columns_.begin + col);
        }

        ConstSubMatrix<T> ToConst() const {
                return ConstSubMatrix<T>(*pmatrix_, {rows_.begin, rows_.end},
                                         {columns_.begin, columns_.end});
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
        ColumnSlice columns_;
};
}// namespace LinAlgTools
