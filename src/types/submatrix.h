#pragma once

#include "matrix.h"
#include "misc.h"

#include <cassert>
#include <cstddef>
#include <vector>

namespace LinAlgTools {
template<typename T>
class SubMatrix {
public:
        SubMatrix(Matrix<T>& matrix, Slice rows, Slice cols)
        {
                assert(rows.begin > -1 && rows.end < matrix.Rows() &&
                       cols.begin > -1 && cols.end < matrix.Columns() &&
                       "Slice must be inside the matrix");
                
                ptr_ = &matrix;
                cols_ = cols.end - cols.begin + 1;
                for (std::size_t i = rows.begin; i <= rows.end; i++) {
                        for (std::size_t j = cols.begin; j <= cols.end; j++) {
                                data_.push_back(matrix(i, j));
                        }
                }
        }

        std::size_t Rows() const
        {
                return data_.size() / cols_;
        }

        std::size_t Columns() const
        {
                return cols_;
        }

        T operator()(const std::size_t row, const std::size_t col) const
        {
                return data_[row * cols_ + col];
        }

        T& operator()(const std::size_t row, const std::size_t col)
        {
                return data_[row * cols_ + col];
        }

        friend std::ostream& operator<<(std::ostream& os, const SubMatrix& matrix)
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

private:
        Matrix<T>* ptr_;
        std::size_t cols_;
        std::vector<T> data_;
};
}// namespace LinAlgTools
