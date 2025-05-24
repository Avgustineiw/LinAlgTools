#pragma once

#include "../core/math_utils.h"
#include "../core/matrix_traits.h"
#include "../core/indices.h"
#include "const_sub_matrix.h"
#include "matrix.h"

#include <cassert>
#include <cstddef>
#include <ostream>
#include <utility>

namespace LinAlgTools {
template<typename T>
class SubMatrix {
        using Index = Core::Indices::Index;
        using RowSlice = Core::Indices::RowSlice;
        using ColumnSlice = Core::Indices::ColumnSlice;

public:
        using ElementType = std::remove_cv_t<T>;

        SubMatrix(Matrix<T>& matrix, RowSlice rows = {0, -1}, ColumnSlice columns = {0, -1})
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

        SubMatrix(const SubMatrix& rhs) = default;

        SubMatrix(SubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, 1})),
              columns_(std::exchange(rhs.columns_, {0, 1})) {};

        SubMatrix(Matrix<T>&& rhs) = delete;

        SubMatrix& operator=(const SubMatrix& lhs) = default;

        SubMatrix& operator=(SubMatrix&& rhs) noexcept {
                pmatrix_ = std::exchange(rhs.pmatrix_, nullptr);
                rows_ = std::exchange(rhs.rows_, {0, 1});
                columns_ = std::exchange(rhs.columns_, {0, 1});
                return *this;
        }

        ConstSubMatrix<T> ToConstSubMatrix() const {
                return ConstSubMatrix<T>(*pmatrix_,
                                         {rows_.begin, rows_.end},
                                         {columns_.begin, columns_.end});
        }

        SubMatrix GetSubMatrix(RowSlice rows, ColumnSlice columns) {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(rows_.begin + rows.begin < Rows() &&
                       rows_.begin + rows.end <= Rows() &&
                       columns_.begin + columns.begin < Columns() &&
                       columns_.begin + columns.end <= Columns() &&
                       "Slice must be inside the matrix.");

                return SubMatrix(*pmatrix_,
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

        SubMatrix GetRow(Index row) {
                assert(pmatrix_ != nullptr &&
                       "Pointer is null");
                assert(row > 0 && row <= Rows() &&
                       "Incorrect row index.");
                return SubMatrix(*pmatrix_,
                                 {rows_.begin + row - 1, rows_.begin + row - 1},
                                 {columns_.begin, columns_.end});
        }

        SubMatrix GetColumn(Index column) {
                assert(pmatrix_ != nullptr &&
                       "Pointer is null.");
                assert(column > 0 && column <= Columns() &&
                       "Incorrect column index.");
                return SubMatrix(*pmatrix_,
                                 {rows_.begin, rows_.end},
                                 {columns_.begin + column - 1, columns_.begin + column - 1});
        }

        Matrix<T> Diagonal() const {
                return ToConstSubMatrix().Diagonal();
        }

        T Trace() const {
                return ToConstSubMatrix().Trace();
        }

        T GetVector2Norm() const {
                return ToConstSubMatrix().GetVector2Norm();
        }

        T GetFrobeniusNorm() const {
                return ToConstSubMatrix().GetFrobeniusNorm();
        }


        SubMatrix& Normalize() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                T norm = GetVector2Norm();
                if (norm != T{0}) {
                        *this /= norm;
                }
                RemoveZeros();
                return *this;
        }

        SubMatrix& Transpose() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                transposed_ = true;
                SwapRowsColumns(rows_, columns_);
                return *this;
        }

        Matrix<T> Transposed() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                Matrix<T> result{*this};
                result.Transpose();
                return result;
        }

        SubMatrix& ConjugateTranspose() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                if constexpr (Core::IsComplexType<T>) {
                        Elementwise([](T& value) {
                                value = std::conj(value);
                        });
                }
                transposed_ = true;
                SwapRowsColumns(rows_, columns_);
                return *this;
        }

        Matrix<T> ConjugateTransposed() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                Matrix<T> result{*this};
                if constexpr (Core::IsComplexType<T>) {
                        result.Elementwise([](T& value) {
                                value = std::conj(value);
                        });
                }
                result.Transpose();
                return *this;
        }


        template<class Function>
        SubMatrix& Elementwise(Function function) {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                function((*this)(i, j));
                        }
                }
                return *this;
        }

        template<class Function>
        const ConstSubMatrix<T>& Elementwise(Function function) const {
                return ToConstSubMatrix().Elementwise(function);
        }

        SubMatrix& RemoveZeros() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                Elementwise([](T& value) {
                        if (Core::IsZero(value)) {
                                value = 0;
                        };
                });
                return *this;
        }

        T operator()(Index row, Index column) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                if (transposed_) {
                        return (*pmatrix_)(columns_.begin + column, rows_.begin + row);
                }
                return (*pmatrix_)(rows_.begin + row, columns_.begin + column);
        }

        T& operator()(const Index row, const Index column) {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                if (transposed_) {
                        return (*pmatrix_)(columns_.begin + column, rows_.begin + row);
                }

                return (*pmatrix_)(rows_.begin + row, columns_.begin + column);
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
        bool transposed_ = false;

        void SwapRowsColumns(RowSlice& rows, ColumnSlice& columns) {
                rows.begin ^= columns.begin;
                columns.begin ^= rows.begin;
                rows.begin ^= columns.begin;

                rows.end ^= columns.end;
                columns.end ^= rows.end;
                rows.end ^= columns.end;
        }
};
}// namespace LinAlgTools

