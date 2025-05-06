#pragma once


#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace LinAlgTools {
template<typename T> class Matrix;
template<typename T> class SubMatrix;

template<typename T>
class ConstSubMatrix {
        using Index = int64_t;

        struct RowSlice
        {
                Index begin = -1;
                Index end = -1;
        };
        struct ColumnSlice
        {
                Index begin = -1;
                Index end = -1;
        };

public:
        ConstSubMatrix(const Matrix<T>& matrix, RowSlice rows, ColumnSlice cols)
            : pmatrix_(&matrix), rows_({rows.begin, rows.end}), cols_({cols.begin, cols.end}) {
                assert(rows.begin > -1 && rows.end < matrix.Rows() &&
                       cols.begin > -1 && cols.end < matrix.Columns() &&
                       "Slice must be inside the matrix");
        }

        Index Rows() const {
                return rows_.end - rows_.begin + 1;
        }

        Index Columns() const {
                return cols_.end - cols_.begin + 1;
        }

        T operator()(Index row, Index col) const {
                assert(pmatrix_ != nullptr && "Pointer is null");
                return (*pmatrix_)(rows_.begin + row, cols_.begin + col);
        }


        friend std::ostream& operator<<(std::ostream& os, const ConstSubMatrix& matrix) {
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
        const Matrix<T>* pmatrix_;
        RowSlice rows_;
        ColumnSlice cols_;
};
}//namespace LinAlgTools
