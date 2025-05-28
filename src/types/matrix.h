#pragma once

#include "../core/indices.h"
#include "../core/is_complex.h"
#include "../core/matrix_traits.h"
#include "const_sub_matrix.h"
#include "sub_matrix.h"

#include <cassert>
#include <complex>
#include <initializer_list>
#include <ostream>
#include <utility>
#include <vector>

namespace LinAlgTools {
using Index = Core::Indices::Index;

template<typename T>
class Matrix {
        using RowSlice = Core::Indices::RowSlice;
        using ColumnSlice = Core::Indices::ColumnSlice;

public:
        using ElementType = std::remove_cv_t<T>;

        explicit Matrix(Index size) : columns_(size), data_(size * size, T{0}) {};

        Matrix(Index rows, Index columns, T value = T{0}) : columns_(columns), data_(rows * columns, value) {
                assert(rows > 0 && columns > 0 &&
                       "Rows and columns must be positive integers.");
        }

        Matrix(std::initializer_list<std::initializer_list<T>> list) : columns_(list.begin()->size()) {
                for (auto sublist: list) {
                        assert(sublist.size() == columns_ &&
                               "Size of rows must be equal to the number of columns.");
                        data_.insert(data_.end(), sublist);
                }
        }

        Matrix(const ConstSubMatrix<T>& rhs) : Matrix(rhs.Rows(), rhs.Columns()) {
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                (*this)(i, j) = rhs(i, j);
                        }
                }
        }

        Matrix(const SubMatrix<T>& rhs) : Matrix(rhs.ToConstSubMatrix()) {};

        Matrix(const Matrix& rhs) = default;

        Matrix(Matrix&& rhs) noexcept
            : columns_(std::exchange(rhs.columns_, 0)), data_(std::move(rhs.data_)) {
        }

        Matrix& operator=(const Matrix& rhs) = default;

        Matrix& operator=(Matrix&& rhs) noexcept {
                columns_ = std::exchange(rhs.columns_, 0);
                data_ = std::move(rhs.data_);
                return *this;
        }

        SubMatrix<T> ToSubMatrix() {
                return SubMatrix<T>(*this);
        }

        ConstSubMatrix<T> ToSubMatrix() const {
                return ConstSubMatrix<T>(*this);
        }

        SubMatrix<T> GetSubMatrix(RowSlice rows, ColumnSlice columns) {
                return ToSubMatrix().GetSubMatrix({rows.begin, rows.end},
                                                  {columns.begin, columns.end});
        }

        ConstSubMatrix<T> GetSubMatrix(RowSlice rows, ColumnSlice columns) const {
                return ToSubMatrix().GetSubMatrix({rows.begin, rows.end},
                                                  {columns.begin, columns.end});
        }

        Index Rows() const {
                return data_.size() / columns_;
        }

        Index Columns() const {
                return columns_;
        }

        SubMatrix<T> GetRow(Index row) {
                return ToSubMatrix().GetRow(row);
        }

        ConstSubMatrix<T> GetRow(Index row) const {
                return ToSubMatrix().GetRow(row);
        }

        SubMatrix<T> GetColumn(Index column) {
                return ToSubMatrix().GetColumn(column);
        }

        ConstSubMatrix<T> GetColumn(Index column) const {
                return ToSubMatrix().GetColumn(column);
        }

        Matrix Diagonal() const {
                return ToSubMatrix().Diagonal();
        }

        T Trace() const {
                return ToSubMatrix().Trace();
        }

        Matrix& Transpose() {
                Matrix result(Columns(), Rows());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                result(j, i) = (*this)(i, j);
                        }
                }
                *this = std::move(result);
                return *this;
        }

        Matrix Transposed() const {
                return ToSubMatrix().Transposed();
        }

        Matrix& ConjugateTranspose() {
                if constexpr (Core::IsComplexType<T>) {
                        Elementwise([](T& value) {
                                value = std::conj(value);
                        });
                }
                Transpose();
                return *this;
        }

        Matrix ConjugateTransposed() const {
                return ToSubMatrix().ConjugateTransposed();
        }


        template<class Function>
        Matrix& Elementwise(Function function) {
                ToSubMatrix().Elementwise(function);
                return *this;
        }

        template<class Function>
        const Matrix& Elementwise(Function function) const {
                ToSubMatrix().Elementwise(function);
                return *this;
        }

        Matrix& RemoveZeros() {
                ToSubMatrix().RemoveZeros();
                return *this;
        }

        double GetVector2Norm() const {
                return ToSubMatrix().GetVector2Norm();
        }

        double GetFrobeniusNorm() const {
                return ToSubMatrix().GetFrobeniusNorm();
        }

        Matrix& Normalize() {
                ToSubMatrix().Normalize();
                return *this;
        }

        static Matrix Identity(Index size) {
                Matrix res(size, size, 0);
                for (int i = 0; i < size; i++) {
                        res(i, i) = 1;
                }
                return res;
        }

        T operator()(const Index row_id, const Index column_id) const {
                return data_[row_id * columns_ + column_id];
        }

        T& operator()(const Index row_id, const Index column_id) {
                return data_[row_id * columns_ + column_id];
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
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
        Index columns_;
        std::vector<T> data_;
};

template<Core::MatrixType F, Core::MatrixType S>
Matrix<typename F::ElementType> operator+(const F& lhs, const S& rhs) {
        using T = typename F::ElementType;

        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for sum.");

        Matrix<T> result = lhs;
        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        result(i, j) += rhs(i, j);
                }
        }

        return result;
}

template<Core::MutableMatrixType F, Core::MatrixType S>
F& operator+=(F& lhs, const S& rhs) {
        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for sum.");

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        lhs(i, j) += rhs(i, j);
                }
        }

        return lhs;
}

template<Core::MatrixType F, Core::MatrixType S>
Matrix<typename F::ElementType> operator-(const F& lhs, const S& rhs) {
        using T = typename F::ElementType;

        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for subtraction.");

        Matrix<T> result = lhs;
        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        result(i, j) -= rhs(i, j);
                }
        }

        return result;
}

template<Core::MutableMatrixType F, Core::MatrixType S>
F& operator-=(F& lhs, const S& rhs) {
        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for sum.");

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        lhs(i, j) -= rhs(i, j);
                }
        }

        return lhs;
}

template<Core::MatrixType F, Core::MatrixType S>
Matrix<typename F::ElementType> operator*(const F& lhs, const S& rhs) {
        using T = typename F::ElementType;

        assert(lhs.Columns() == rhs.Rows() &&
               "Number of columns of the left matrix must equal the number of rows of the right matrix.");

        Matrix<T> result(lhs.Rows(), rhs.Columns());

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < rhs.Columns(); j++) {
                        T sum = 0;
                        for (Index k = 0; k < lhs.Columns(); k++) {
                                sum += lhs(i, k) * rhs(k, j);
                        }
                        result(i, j) = sum;
                }
        }

        result.RemoveZeros();
        return result;
}

template<Core::MutableMatrixType F, Core::MatrixType S>
F& operator*=(F& lhs, const S& rhs) {
        if (lhs.Rows() == 0 || rhs.Rows() == 0) {
                return lhs;
        }

        assert(lhs.Columns() == rhs.Rows() && rhs.Rows() == rhs.Columns() &&
               "Number of columns of the left matrix must equal the number of rows of the right matrix");

        auto result = lhs * rhs;
        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        lhs(i, j) = result(i, j);
                }
        }

        result.RemoveZeros();
        return lhs;
}

template<Core::MatrixType F>
Matrix<typename F::ElementType> operator*(const F& lhs, typename F::ElementType scalar) {
        using T = typename F::ElementType;
        Matrix<T> result = lhs;

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        result(i, j) *= scalar;
                }
        }

        result.RemoveZeros();
        return result;
}

template<Core::MatrixType F>
Matrix<typename F::ElementType> operator*(typename F::ElementType scalar, const F& rhs) {
        return rhs * scalar;
}

template<Core::MutableMatrixType F>
F& operator*=(F& lhs, typename F::ElementType scalar) {
        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        lhs(i, j) *= scalar;
                }
        }

        lhs.RemoveZeros();
        return lhs;
}

template<Core::MatrixType F>
Matrix<typename F::ElementType> operator/(const F& lhs, typename F::ElementType scalar) {
        using T = typename F::ElementType;
        Matrix<T> result = lhs;

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        result(i, j) /= scalar;
                }
        }

        result.RemoveZeros();
        return result;
}

template<Core::MatrixType F>
Matrix<typename F::ElementType> operator/(typename F::ElementType scalar, const F& rhs) {
        return rhs / scalar;
}

template<Core::MutableMatrixType F>
F& operator/=(F& lhs, typename F::ElementType scalar) {
        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        lhs(i, j) /= scalar;
                }
        }

        lhs.RemoveZeros();
        return lhs;
}

template<Core::MatrixType F, Core::MatrixType S>
bool operator==(const F& lhs, const S& rhs) {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns()) {
                return false;
        }

        for (Index i = 0; i < lhs.Rows(); i++) {
                for (Index j = 0; j < lhs.Columns(); j++) {
                        if (lhs(i, j) != rhs(i, j)) {
                                return false;
                        }
                }
        }

        return true;
}

template<Core::MatrixType F, Core::MatrixType S>
bool operator!=(const F& lhs, const S& rhs) {
        return !(lhs == rhs);
}
}//namespace LinAlgTools

