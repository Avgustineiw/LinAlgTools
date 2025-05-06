#pragma once

#include "../helpers/index.h"
#include "../helpers/matrix_type.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

namespace LinAlgTools {
template<typename T>
class SubMatrix;
template<typename T>
class ConstSubMatrix;

template<typename T>
class Matrix {
        using Index = Helpers::Types::Index;

public:
        using ElementType = T;

        Matrix(Index rows, Index cols, T value = T{0}) : cols_(cols), data_(rows * cols, value) {
                assert(rows > 0 && cols > 0 &&
                       "Rows and columns must be positive integers.");
        }

        Matrix(std::initializer_list<std::initializer_list<T>> list) : cols_(list.begin()->size()) {
                for (auto sublist: list) {
                        assert(sublist.size() == cols_ && 
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

        Matrix(const SubMatrix<T>& rhs) : Matrix(rhs.ToConst()) {};

        Index Rows() const {
                return data_.size() / cols_;
        }

        Index Columns() const {
                return cols_;
        }

        Matrix Diagonal() const {
                const Index size = std::min(Rows(), Columns());
                Matrix<T> result(size, 1);
                for (Index i = 0; i < size; i++) {
                        result(i, 0) = (*this)(i, i);
                }
                return result;
        }

        T Trace() const {
                T result = 0;
                const Index size = std::min(Rows(), Columns());
                for (Index i = 0; i < size; i++) {
                        result += (*this)(i, i);
                }
                return result;
        }

        Matrix& Transpose() {
                Matrix<T> result(Columns(), Rows());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                result(j, i) = (*this)(i, j);
                        }
                }
                *this = std::move(result);
                return *this;
        }

        Matrix Transposed() {
                Matrix<T> result(Columns(), Rows());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                result(j, i) = (*this)(i, j);
                        }
                }
                return result;
        }

        template<class Function>
        Matrix& Elementwise(Function f) {
                for (auto& element: data_) {
                        f(element);
                }
                return *this;
        }

        template<class Function>
        const Matrix& Elementwise(Function f) const {
                for (const auto& element: data_) {
                        f(element);
                }
                return *this;
        }

        T GetNorm() const {
                assert(Rows() == 1 || Columns() == 1 && 
                       "Incorrect size for vector norm");

                T res = T{0};
                Elementwise([&res](const T& value) {
                        res += value * value;
                });
                return std::sqrt(res);
        }

        Matrix& Normalize() {
                T norm = GetNorm();
                if (norm != 0) {
                        *this = *this / norm;
                }
                return *this;
        }

        static Matrix Identity(Index size) {
                Matrix res(size, size, 0);
                for (int i = 0; i < size; i++) {
                        res(i, i) = 1;
                }
                return res;
        }

        T operator()(const Index row_id, const Index col_id) const {
                return data_[row_id * cols_ + col_id];
        }

        T& operator()(const Index row_id, const Index col_id) {
                return data_[row_id * cols_ + col_id];
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
        Index cols_;
        std::vector<T> data_;
};

using Index = Helpers::Types::Index;

template<Helpers::MatrixType F, Helpers::MatrixType S>
Matrix<typename F::ElementType> operator+(const F& lhs, const S& rhs) {
        using T = typename F::ElementType;

        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for sum.");

        Matrix<T> result = lhs;
        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        result(i, j) += rhs(i, j);
                }
        }

        return result;
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
F& operator+=(F& lhs, const S& rhs) {
        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for sum.");

        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        lhs(i, j) += rhs(i, j);
                }
        }

        return lhs;
}

template<Helpers::MatrixType F, Helpers::MatrixType S>
Matrix<typename F::ElementType> operator-(const F& lhs, const S& rhs) {
        using T = typename F::ElementType;

        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() && 
               "Matrices must have the same size for subtraction.");

        Matrix<T> result = lhs;
        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        result(i, j) -= rhs(i, j);
                }
        }

        return result;
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
F& operator-=(F& lhs, const S& rhs) {
        assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
               "Matrices must have the same size for sum.");

        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        lhs(i, j) -= rhs(i, j);
                }
        }

        return lhs;
}

template<Helpers::MatrixType F, Helpers::MatrixType S>
Matrix<typename F::ElementType> operator*(const F& lhs, const S& rhs) {
        using T = typename F::ElementType;

        assert(lhs.Columns() == rhs.Rows()  && 
               "Number of columns of the left matrix must equal the number of rows of the right matrix.");

        Matrix<T> result(lhs.Rows(), rhs.Columns());

        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < rhs.Columns(); ++j) {
                        T sum = 0;
                        for (Index k = 0; k < lhs.Columns(); ++k) {
                                sum += lhs(i, k) * rhs(k, j);
                        }
                        result(i, j) = sum;
                }
        }

        return result;
}

template<Helpers::MutableMatrixType F, Helpers::MatrixType S>
F& operator*=(F& lhs, const S& rhs) {
        if (lhs.Rows() == 0 || rhs.Rows() == 0) {
                return lhs;
        }

        assert(lhs.Columns() == rhs.Rows() && rhs.Rows() == rhs.Columns() &&
               "Number of columns of the left matrix must equal the number of rows of the right matrix");

        auto result = lhs * rhs;
        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        lhs(i, j) = result(i, j);
                }
        }

        return lhs;
}

template<Helpers::MatrixType F>
Matrix<typename F::ElementType> operator*(const F& lhs, typename F::ElementType scalar) {
        using T = typename F::ElementType;
        Matrix<T> result = lhs;

        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        result(i, j) *= scalar;
                }
        }

        return result;
}

template<Helpers::MatrixType F>
Matrix<typename F::ElementType> operator*(typename F::ElementType scalar, const F& rhs) {
        return rhs * scalar;
}

template<Helpers::MutableMatrixType F>
F& operator*=(F& lhs, typename F::ElementType scalar) {
        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        lhs(i, j) *= scalar;
                }
        }

        return lhs;
}

template<Helpers::MatrixType F>
Matrix<typename F::ElementType> operator/(const F& lhs, typename F::ElementType scalar) {
        using T = typename F::ElementType;
        Matrix<T> result = lhs;

        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        result(i, j) /= scalar;
                }
        }

        return result;
}

template<Helpers::MatrixType F>
Matrix<typename F::ElementType> operator/(typename F::ElementType scalar, const F& rhs) {
        return rhs / scalar;
}

template<Helpers::MutableMatrixType F>
F& operator/=(F& lhs, typename F::ElementType scalar) {
        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        lhs(i, j) /= scalar;
                }
        }

        return lhs;
}

template<Helpers::MatrixType F, Helpers::MatrixType S>
bool operator==(const F& lhs, const S& rhs) {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns()) {
                return false;
        }

        for (Index i = 0; i < lhs.Rows(); ++i) {
                for (Index j = 0; j < lhs.Columns(); ++j) {
                        if (lhs(i, j) != rhs(i, j)) {
                                return false;
                        }
                }
        }

        return true;
}

template<Helpers::MatrixType F, Helpers::MatrixType S>
bool operator!=(const F& lhs, const S& rhs) {
        return !(lhs == rhs);
}
}//namespace LinAlgTools
