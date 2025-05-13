#pragma once

#include "../helpers/matrix_type.h"
#include "../helpers/types.h"
#include "matrix.h"
#include "submatrix.h"

#include <cassert>
#include <cstddef>
#include <ostream>

namespace LinAlgTools {
template<typename T>
class ConstSubMatrix {
        using Index = Helpers::Types::Index;
        using RowSlice = Helpers::Types::RowSlice;
        using ColumnSlice = Helpers::Types::ColumnSlice;

public:
        using ElementType = std::remove_cv_t<T>;

        ConstSubMatrix(const Matrix<T>& matrix, RowSlice rows = {0, -1}, ColumnSlice columns = {0, -1})
            : pmatrix_(&matrix),
              rows_({rows.begin, rows.end}),
              columns_({columns.begin, columns.end}) {
                assert(rows.begin > -1 && rows.end < matrix.Rows() &&
                       columns.begin > -1 && columns.end < matrix.Columns() &&
                       "Slice must be inside the matrix.");

                if (rows.end == -1) {
                        rows_.end = matrix.Rows() - 1;
                }
                if (columns.end == -1) {
                        columns_.end = matrix.Columns() - 1;
                }
        }

        ConstSubMatrix(const ConstSubMatrix& rhs) = default;

        ConstSubMatrix(ConstSubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, 1})),
              columns_(std::exchange(rhs.columns_, {0, 1})) {};

        ConstSubMatrix(Matrix<T>&& rhs) = delete;

        ConstSubMatrix& operator=(const ConstSubMatrix& lhs) = default;

        ConstSubMatrix& operator=(ConstSubMatrix&& rhs) noexcept {
                pmatrix_ = std::exchange(rhs.pmatrix_, nullptr);
                rows_ = std::exchange(rhs.rows_, {0, 1});
                columns_ = std::exchange(rhs.columns_, {0, 1});
                return *this;
        }

        ConstSubMatrix<T> GetConstSubMatrix(RowSlice rows, ColumnSlice columns) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(rows_.begin + rows.begin < Rows() &&
                       rows_.begin + rows.end <= Rows() &&
                       columns_.begin + columns.begin < Columns() &&
                       columns_.begin + columns.end <= Columns() &&
                       "Slice must be inside the matrix.");

                return ConstSubMatrix<T>(*pmatrix_,
                                         {rows_.begin + rows.begin, rows_.begin + rows.end},
                                         {columns_.begin + columns.begin, columns_.begin + columns.end});
        }


        Index Rows() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                return rows_.end - rows_.begin + 1;
        }

        Index Columns() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                return columns_.end - columns_.begin + 1;
        }

        ConstSubMatrix<T> GetRow(Index row) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(row > 0 && row <= Rows() &&
                       "Incorrect row index.");
                return ConstSubMatrix(*pmatrix_,
                                      {rows_.begin + row - 1, rows_.begin + row - 1},
                                      {columns_.begin, columns_.end});
        }

        ConstSubMatrix<T> GetColumn(Index column) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(column > 0 && column <= Columns() &&
                       "Incorrect column index.");
                return ConstSubMatrix(*pmatrix_,
                                      {rows_.begin, rows_.end},
                                      {columns_.begin + column - 1, columns_.begin + column - 1});
        }

        template<class Function>
        const ConstSubMatrix& Elementwise(Function function) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                function((*this)(i, j));
                        }
                }
                return *this;
        }

        Matrix<T> Diagonal() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                const Index size = std::min(Rows(), Columns());
                Matrix<T> result(size, 1);
                for (Index i = 0; i < size; i++) {
                        result(i, 0) = (*this)(i, i);
                }
                return result;
        }

        T Trace() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                T result = 0;
                const Index size = std::min(Rows(), Columns());
                for (Index i = 0; i < size; i++) {
                        result += (*this)(i, i);
                }
                return result;
        }

        T Get2Norm() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(Rows() == 1 || Columns() == 1 &&
                                              "Incorrect size for vector norm.");

                T res = T{0};
                Elementwise([&res](const T& value) {
                        res += value * value;
                });
                return std::sqrt(res);
        }

        Matrix<T> Transposed() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                Matrix<T> result{*this};
                result.Transpose();
                return result;
        }

        T operator()(Index row, Index column) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                return (*pmatrix_)(rows_.begin + row, columns_.begin + column);
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
        ColumnSlice columns_;
        bool transposed_ = false;
};
}//namespace LinAlgTools
