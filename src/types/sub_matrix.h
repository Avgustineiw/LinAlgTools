#pragma once

#include "../core/indices.h"
#include "../core/math_utils.h"
#include "../core/matrix_traits.h"
#include "const_sub_matrix.h"
#include "matrix.h"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <utility>

namespace LinAlgTools {
template<typename T>
class SubMatrix {
        friend class LinAlgTools::ConstSubMatrix<T>;
        using Index = Core::Indices::Index;
        using Slice = Core::Indices::Slice;

public:
        using ElementType = std::remove_cv_t<T>;

        /*
        * Slicing uses inclusive bounds [first, last], numbering from 0
        * Columns = {0, 1} will include both columns 0 and 1
        * -1 in the second slot indicates the amount of rows or columns
        * Rows = {0, -1} will get all rows of the matrix
        */
        SubMatrix(Matrix<T>& matrix, Slice rows = {0, -1}, Slice columns = {0, -1})
            : pmatrix_(&matrix),
              rows_({rows.first, rows.last}),
              columns_({columns.first, columns.last}) {
                assert(rows_.first > -1 && rows_.first < matrix.Rows() &&
                       rows_.last >= -1 && rows_.last < matrix.Rows() &&
                       columns_.first > -1 && columns_.first < matrix.Columns() &&
                       columns_.last >= -1 && columns_.last < matrix.Columns() &&
                       "Slice must be inside the matrix.");

                if (rows.last == -1) {
                        rows_.last = matrix.Rows() - 1;
                }
                if (columns.last == -1) {
                        columns_.last = matrix.Columns() - 1;
                }

                assert(rows_.first <= rows_.last &&
                       columns_.first <= columns_.last);
        }

        SubMatrix(const SubMatrix& rhs) = default;

        explicit SubMatrix(const ConstSubMatrix<T>& rhs)
            : pmatrix_(const_cast<Matrix<T>*>(rhs.pmatrix_)),
              rows_(rhs.rows_),
              columns_(rhs.columns_) {
                assert(rhs.pmatrix_ != nullptr &&
                       "Matrix pointer is null");
        }

        SubMatrix(SubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, -1})),
              columns_(std::exchange(rhs.columns_, {0, -1})) {}

        SubMatrix(Matrix<T>&& rhs) = delete;

        ConstSubMatrix<T> ToConstSubMatrix() const {
                return {*pmatrix_,
                        {rows_.first, rows_.last},
                        {columns_.first, columns_.last}};
        }

        ConstSubMatrix<T> ToConstSubMatrix() && = delete;

        SubMatrix& operator=(const SubMatrix&) = default;

        SubMatrix& operator=(SubMatrix&& rhs) noexcept {
                pmatrix_ = std::exchange(rhs.pmatrix_, nullptr);
                rows_ = std::exchange(rhs.rows_, {0, -1});
                columns_ = std::exchange(rhs.columns_, {0, -1});
                return *this;
        }

        SubMatrix GetSubMatrix(Slice rows, Slice columns) {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(rows_.first + rows.first < Rows() &&
                       rows_.first + rows.last <= Rows() &&
                       columns_.first + columns.first < Columns() &&
                       columns_.first + columns.last <= Columns() &&
                       "Slice must be inside the matrix.");

                return SubMatrix(*pmatrix_,
                                 {rows_.first + rows.first, rows_.first + rows.last},
                                 {columns_.first + columns.first, columns_.first + columns.last});
        }

        Index Rows() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                return rows_.last - rows_.first + 1;
        }

        Index Columns() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                return columns_.last - columns_.first + 1;
        }

        SubMatrix GetRow(Index row) {
                assert(pmatrix_ != nullptr &&
                       "Pointer is null");
                assert(row >= 0 && row < Rows() &&
                       "Incorrect row index.");
                return SubMatrix(*pmatrix_,
                                 {rows_.first + row, rows_.first + row},
                                 {columns_.first, columns_.last});
        }

        SubMatrix GetColumn(Index column) {
                assert(pmatrix_ != nullptr &&
                       "Pointer is null.");
                assert(column >= 0 && column < Columns() &&
                       "Incorrect column index.");
                return SubMatrix(*pmatrix_,
                                 {rows_.first, rows_.last},
                                 {columns_.first + column, columns_.first + column});
        }

        Matrix<T> Diagonal() const {
                return ToConstSubMatrix().Diagonal();
        }

        T Trace() const {
                return ToConstSubMatrix().Trace();
        }

        double GetVector2Norm() const {
                return ToConstSubMatrix().GetVector2Norm();
        }

        double GetFrobeniusNorm() const {
                return ToConstSubMatrix().GetFrobeniusNorm();
        }


        SubMatrix& NormalizeVector() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                T norm = static_cast<T>(GetVector2Norm());
                if (!Core::IsZero(norm)) {
                        *this /= norm;
                }
                RemoveZeros();
                return *this;
        }

        SubMatrix& Transpose() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                Matrix<T> temp = *this;
                *pmatrix_ = std::move(Transposed(temp));
                std::swap(rows_, columns_);
                return *this;
        }

        SubMatrix& ConjugateTranspose() {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");

                if constexpr (Core::IsComplexType<T>) {
                        Elementwise([](T& value) {
                                value = std::conj(value);
                        });
                }
                return (*this).Transpose();
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
                                value = T{0};
                        };
                });
                return *this;
        }

        T operator()(Index row, Index column) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(row >= 0 && row < Rows() &&
                       "Invalid row index");
                assert(column >= 0 && column < Columns() &&
                       "Invalid column index");

                return (*pmatrix_)(rows_.first + row, columns_.first + column);
        }

        T& operator()(const Index row, const Index column) {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(row >= 0 && row < Rows() &&
                       "Invalid row index");
                assert(column >= 0 && column < Columns() &&
                       "Invalid column index");

                return (*pmatrix_)(rows_.first + row, columns_.first + column);
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
        Slice rows_;
        Slice columns_;
};
}// namespace LinAlgTools

