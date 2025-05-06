#pragma once

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
        using Index = int64_t;

public:
        Matrix(Index rows, Index cols, T value = T{0}) : cols_(cols), data_(rows * cols, value) {
                assert(rows > 0 && cols > 0 && "Rows and columns must be positive integers");
        }

        Matrix(std::initializer_list<std::initializer_list<T>> list) : cols_(list.begin()->size()) {
                for (auto sublist: list) {
                        assert(sublist.size() == cols_ && "Rows have different sizes");
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
                assert(Rows() == 1 || Columns() == 1 && "Incorrect size for vector norm");

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

        Matrix& operator+=(const Matrix& rhs) {
                assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() && "Matrices must be of the same size");

                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] += rhs.data_[i];
                }
                return *this;
        }

        friend Matrix operator+(const Matrix& first, const Matrix& second) {
                Matrix result = first;
                result += second;
                return result;
        }
        friend Matrix operator+(Matrix&& first, const Matrix& second) {
                first += second;
                return std::move(first);
        }
        friend Matrix operator+(const Matrix& first, Matrix&& second) {
                second += first;
                return std::move(second);
        }
        friend Matrix operator+(Matrix&& first, Matrix&& second) {
                first += second;
                return std::move(first);
        }

        Matrix& operator-=(const Matrix& rhs) {
                assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() && "Matrices must be of the same size");

                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] -= rhs.data_[i];
                }
                return *this;
        }

        friend Matrix operator-(const Matrix& first, const Matrix& second) {
                Matrix result = first;
                result -= second;
                return result;
        }
        friend Matrix operator-(Matrix&& first, const Matrix& second) {
                first -= second;
                return first;
        }
        friend Matrix operator-(const Matrix& first, Matrix&& second) {
                second -= first;
                return second;
        }
        friend Matrix operator-(Matrix&& first, Matrix&& second) {
                first -= second;
                return first;
        }

        Matrix& operator*=(const Matrix& rhs) {
                assert(Columns() == rhs.Rows() && "Number of columns of the left matrix must equal the number of rows of the right matrix");

                Matrix result = {Rows(), rhs.Columns(), T{0}};
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < rhs.Columns(); j++) {
                                for (Index k = 0; k < Columns(); k++) {
                                        result(i, j) += (*this)(i, k) * rhs(k, j);
                                }
                        }
                }
                *this = std::move(result);
                return *this;
        }

        friend Matrix operator*(const Matrix& first, const Matrix& second) {
                Matrix result = first;
                result *= second;
                return result;
        }
        friend Matrix operator*(Matrix&& first, const Matrix& second) {
                first *= second;
                return first;
        }
        friend Matrix operator*(const Matrix& first, Matrix&& second) {
                second *= first;
                return second;
        }
        friend Matrix operator*(Matrix&& first, Matrix&& second) {
                first *= second;
                return first;
        }

        Matrix& operator*=(T scalar) {
                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] *= scalar;
                }
                return *this;
        }

        friend Matrix operator*(const Matrix& left, T value) {
                Matrix result = left;
                result *= value;
                return result;
        }
        friend Matrix operator*(Matrix&& left, T value) {
                left *= value;
                return left;
        }
        friend Matrix operator*(T value, const Matrix& right) {
                Matrix result = right;
                result *= value;
                return result;
        }
        friend Matrix operator*(T value, Matrix&& right) {
                right *= value;
                return right;
        }

        Matrix& operator/=(T scalar) {
                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] /= scalar;
                }
                return *this;
        }

        friend Matrix operator/(const Matrix& left, T value) {
                Matrix result = left;
                result /= value;
                return result;
        }
        friend Matrix operator/(Matrix&& left, T value) {
                left /= value;
                return left;
        }
        friend Matrix operator/(T value, const Matrix& right) {
                Matrix result = right;
                result /= value;
                return result;
        }
        friend Matrix operator/(T value, Matrix&& right) {
                right /= value;
                return right;
        }

        friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
                return lhs.cols_ == rhs.cols_ && lhs.data_ == rhs.data_;
        }

        bool operator!=(const Matrix& rhs) {
                return !((*this) == rhs);
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
}//namespace LinAlgTools
