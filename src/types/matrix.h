#pragma once

#include <cassert>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace LinAlgTools {
template<typename T>
class Matrix
{
public:
        //constructors
        Matrix(const Matrix& rhs) = default;

        Matrix(std::size_t rows, std::size_t cols, T value = T{0})
        {
                cols_ = cols;

                data_.resize(rows * cols, value);
        }

        Matrix(std::initializer_list<std::initializer_list<T>> list)
        {
                cols_ = list.begin()->size();
                data_.reserve(Columns() * list.size());

                for (auto sublist: list) {
                        for (auto ele: sublist) {
                                data_.push_back(ele);
                        }
                }
        };

        //Methods
        std::size_t Rows() const
        {
                return data_.size() / cols_;
        }

        std::size_t Columns() const
        {
                return cols_;
        }

        Matrix<T> Diagonal() const
        {
                const std::size_t size = std::min(Rows(), Columns());

                Matrix<T> res(size, 1);

                for (std::size_t i = 0; i < size; i++) {
                        res(i, 0) = (*this)(i, i);
                }

                return res;
        }

        T Trace() const
        {
                T res = 0;

                const std::size_t size = std::min(Rows(), Columns());

                for (std::size_t i = 0; i < size; i++) {
                        res += (*this)(i, i);
                }

                return res;
        }

        Matrix<T>& Transpose()
        {
                Matrix<T> res(Columns(), Rows());

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                res(j, i) = (*this)(i, j);
                        }
                }

                *this = res;

                return *this;
        }

        Matrix<T>& Elementwise(std::function<void(T&)> func)
        {
                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                func((*this)(i, j));
                        }
                }

                return *this;
        }

        T GetNorm()
        {
                assert(Rows() == 1 || Columns() == 1 && "Incorrect size");

                T res = T{0};
                Elementwise([&](T& value) { res += value; });
                return res;
        }

        Matrix<T>& Normalize()
        {
                T norm = GetNorm();

                *this = *this / norm;

                return *this;
        }

        //operators
        T operator()(const std::size_t row_id, const std::size_t col_id) const
        {
                return data_[row_id * cols_ + col_id];
        }

        T& operator()(const std::size_t row_id, const std::size_t col_id)
        {
                return data_[row_id * cols_ + col_id];
        }

        Matrix<T>& operator=(const Matrix& rhs) = default;
        // std::size_t rows = rhs.Rows();
        // std::size_t cols = rhs.Columns();
        //
        // assert(rows == Rows() && cols == Columns() && "sizes are different");
        //
        // for (std::size_t i = 0; i < Rows(); i++) {
        //         for (std::size_t j = 0; j < Columns(); j++) {
        //                 (*this)(i, j) = rhs(i, j);
        //         }
        // }
        //
        // return *this;

        Matrix<T> operator+(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && rhs.Columns() && "Matrices must be of the same size");

                Matrix<T> res(Rows(), Columns());

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) + rhs(i, j);
                        }
                }

                return res;
        }

        Matrix<T>& operator+=(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && rhs.Columns() && "Matrices must be of the same size");

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                (*this)(i, j) += rhs(i, j);
                        }
                }

                return *this;
        }

        Matrix<T> operator-(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && rhs.Columns() && "Matrices must be of the same size");

                Matrix<T> res(Rows(), Columns());

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) - rhs(i, j);
                        }
                }

                return res;
        }

        Matrix<T>& operator-=(Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && rhs.Columns() && "Matrices must be of the same size");

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                (*this)(i, j) -= rhs(i, j);
                        }
                }

                return *this;
        }

        Matrix<T> operator*(Matrix& rhs)
        {
                assert(Rows() == rhs.Columns() && "Number of rows of the left matrix must equal the number of columns of the right matrix");

                Matrix<T> res(Rows(), rhs.Columns());

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < rhs.Columns(); j++) {
                                for (std::size_t k = 0; k < Columns(); k++) {
                                        res(i, j) += (*this)(i, k) * rhs(k, j);
                                }
                        }
                }

                return res;
        }

        Matrix<T> operator*(const T scalar)
        {
                Matrix<T> res(Rows(), Columns());

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) * scalar;
                        }
                }

                return res;
        }

        Matrix<T>& operator*=(T scalar)
        {
                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                (*this)(i, j) *= scalar;
                        }
                }

                return *this;
        }

        Matrix<T> operator/(const T scalar)
        {
                Matrix<T> res(Rows(), Columns());

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                res(i, j) = (*this)(i, j) / scalar;
                        }
                }

                return res;
        }

        Matrix<T>& operator/=(T scalar)
        {
                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                (*this)(i, j) /= scalar;
                        }
                }

                return *this;
        }

        bool operator==(const Matrix& rhs)
        {
                assert(Rows() == rhs.Rows() && rhs.Columns() && "Matrices must be of the same size");

                for (std::size_t i = 0; i < Rows(); i++) {
                        for (std::size_t j = 0; j < Columns(); j++) {
                                if ((*this)(i, j) != rhs(i, j)) {
                                        return false;
                                }
                        }
                }

                return true;
        }

        bool operator!=(const Matrix& rhs)
        {
                return !((*this) == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
        {
                for (std::size_t i = 0; i < matrix.Rows(); i++) {
                        os << '[';

                        for (std::size_t j = 0; j < matrix.Columns(); j++) {
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
                for (std::size_t i = 0; i < matrix.Rows(); i++) {
                        for (std::size_t j = 0; j < matrix.Columns(); j++) {
                                is >> matrix(i, j);
                        }
                }

                return is;
        }


private:
        std::size_t cols_;
        std::vector<T> data_;
};
}//namespace LinAlgTools
