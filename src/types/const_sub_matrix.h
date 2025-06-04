#pragma once

#include "../core/indices.h"
#include "../core/matrix_traits.h"
#include "matrix.h"

#include <cassert>
#include <complex>
#include <cstddef>
#include <ostream>

namespace LinAlgTools {
template<typename T>
class ConstSubMatrix {
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
        ConstSubMatrix(const Matrix<T>& matrix, Slice rows = {0, -1}, Slice columns = {0, -1})
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

        ConstSubMatrix(const SubMatrix<T>& rhs)
            : pmatrix_(rhs.pmatrix_),
              rows_(rhs.rows_),
              columns_(rhs.columns_) {
                assert(rhs.pmatrix_ != nullptr 
                       && "Matrix pointer is null");
        }

        ConstSubMatrix(const ConstSubMatrix& rhs) = default;

        ConstSubMatrix(ConstSubMatrix&& rhs) noexcept
            : pmatrix_(std::exchange(rhs.pmatrix_, nullptr)),
              rows_(std::exchange(rhs.rows_, {0, -1})),
              columns_(std::exchange(rhs.columns_, {0, -1})) {}

        ConstSubMatrix(Matrix<T>&& rhs) = delete;

        ConstSubMatrix& operator=(const ConstSubMatrix& lhs) = default;

        ConstSubMatrix& operator=(ConstSubMatrix&& rhs) noexcept {
                pmatrix_ = std::exchange(rhs.pmatrix_, nullptr);
                rows_ = std::exchange(rhs.rows_, {0, -1});
                columns_ = std::exchange(rhs.columns_, {0, -1});
                return *this;
        }

        ConstSubMatrix GetSubMatrix(Slice rows, Slice columns) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(rows_.first + rows.first < Rows() &&
                       rows_.first + rows.last <= Rows() &&
                       columns_.first + columns.first < Columns() &&
                       columns_.first + columns.last <= Columns() &&
                       "Slice must be inside the matrix.");

                return ConstSubMatrix<T>(*pmatrix_,
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

        ConstSubMatrix GetRow(Index row) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(row >= 0 && row < Rows() &&
                       "Incorrect row index.");
                return ConstSubMatrix(*pmatrix_,
                                      {rows_.first + row, rows_.first + row},
                                      {columns_.first, columns_.last});
        }

        ConstSubMatrix GetColumn(Index column) const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(column >= 0 && column < Columns() &&
                       "Incorrect column index.");
                return ConstSubMatrix(*pmatrix_,
                                      {rows_.first, rows_.last},
                                      {columns_.first + column, columns_.first + column});
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

        double GetVector2Norm() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                assert(Rows() == 1 || Columns() == 1 &&
                                              "Incorrect size for vector norm.");
                return Calculate2Norm();
        }

        double GetFrobeniusNorm() const {
                assert(pmatrix_ != nullptr &&
                       "Matrix pointer is null.");
                return Calculate2Norm();
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
        double Calculate2Norm() const {
                double result = 0.0;
                Elementwise([&result](const T& value) {
                        result += std::norm(value);
                });
                return std::sqrt(result);
        }

        const Matrix<T>* pmatrix_;
        Slice rows_;
        Slice columns_;
};
}//namespace LinAlgTools

