#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace LinAlgTools {
template<typename T>
class Matrix {
        using Index = uint64_t;

public:
        Matrix(Index rows, Index cols, T value = T{0}) : cols_(cols), data_(rows * cols, value)
        {//нужно ли тут мувать cols и rows?
                assert(rows > 0 || cols > 0 && "Rows and columns must be positive integers");
        }

        Matrix(std::initializer_list<std::initializer_list<T>> list) : cols_(list.begin()->size())//код не падает, если список пустой
        {
                for (auto sublist: list) {
                        assert(sublist.size() == cols_ && "Rows have different sizes");
                        data_.insert(data_.end(), sublist);
                }
        }

        Index Rows() const
        {
                return data_.size() / cols_;
        }

        Index Columns() const
        {
                return cols_;
        }

        Matrix Diagonal() const
        {
                const Index size = std::min(Rows(), Columns());
                Matrix<T> res(size, 1);
                for (Index i = 0; i < size; i++) {
                        res(i, 0) = (*this)(i, i);
                }

                return res;
        }

        T Trace() const
        {
                T res = 0;
                const Index size = std::min(Rows(), Columns());
                for (Index i = 0; i < size; i++) {
                        res += (*this)(i, i);
                }

                return res;
        }

        Matrix& Transpose()
        {
                Matrix<T> res(Columns(), Rows());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                res(j, i) = (*this)(i, j);
                        }
                }
                *this = std::move(res);

                return *this;
        }

        template<class Function>
        Matrix& Elementwise(Function f)
        {
                for (auto& element: data_) {
                        f(element);
                }

                return *this;
        }

        template<class Function>
        const Matrix& Elementwise(Function f) const
        {//void или const Matrix&?
                for (auto& element: data_) {
                        f(element);
                }

                return *this;
        }

        T GetNorm() const
        {
                assert(Rows() == 1 || Columns() == 1 && "Incorrect size for vector norm");

                T res = T{0};
                Elementwise([&res](const T& value) {
                        res += value * value;
                });

                return std::sqrt(res);
        }

        Matrix& Normalize()
        {
                T norm = GetNorm();
                if (norm != 0) {
                        *this = *this / norm;
                }

                return *this;
        }

        //operators
        T operator()(const Index row_id, const Index col_id) const
        {
                return data_[row_id * cols_ + col_id];
        }

        T& operator()(const Index row_id, const Index col_id)
        {
                return data_[row_id * cols_ + col_id];
        }

        Matrix operator+(const Matrix& rhs) const
        {
                assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() && "Matrices must be of the same size");

                Matrix<T> res(Rows(), Columns());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) + rhs(i, j);
                        }
                }

                return res;
        }

        Matrix& operator+=(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() && "Matrices must be of the same size");

                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] += rhs.data_[i];
                }

                return *this;
        }

        Matrix operator-(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() && "Matrices must be of the same size");

                Matrix<T> res(Rows(), Columns());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) - rhs(i, j);
                        }
                }

                return res;
        }

        Matrix& operator-=(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() && "Matrices must be of the same size");

                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] -= rhs.data_[i];
                }

                return *this;
        }

        Matrix operator*(Matrix& rhs)
        {
                assert(Rows() == rhs.Columns() && "Number of rows of the left matrix must equal the number of columns of the right matrix");

                Matrix<T> res(Rows(), rhs.Columns());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < rhs.Columns(); j++) {
                                for (Index k = 0; k < Columns(); k++) {
                                        res(i, j) += (*this)(i, k) * rhs(k, j);
                                }
                        }
                }

                return res;
        }

        Matrix operator*(const T scalar)
        {
                Matrix<T> res(Rows(), Columns());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) * scalar;
                        }
                }

                return res;
        }

        Matrix& operator*=(T scalar)
        {
                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] *= scalar;
                }

                return *this;
        }

        Matrix operator/(const T scalar)
        {
                Matrix<T> res(Rows(), Columns());
                for (Index i = 0; i < Rows(); i++) {
                        for (Index j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) / scalar;
                        }
                }

                return res;
        }

        Matrix& operator/=(T scalar)
        {
                for (Index i = 0; i < data_.size(); i++) {
                        data_[i] /= scalar;
                }

                return *this;
        }

        friend bool operator==(const Matrix& lhs, const Matrix& rhs)
        {
                return lhs.cols_ == rhs.cols_ && lhs.data_ == rhs.data_;
        }

        bool operator!=(const Matrix& rhs)
        {
                return !((*this) == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
        {
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

        friend std::istream& operator>>(std::istream& is, Matrix& matrix)
        {
                for (Index i = 0; i < matrix.Rows(); i++) {
                        for (Index j = 0; j < matrix.Columns(); j++) {
                                is >> matrix(i, j);
                        }
                }

                return is;
        }


private:
        Index cols_;
        std::vector<T> data_;
};
}//namespace LinAlgTools
