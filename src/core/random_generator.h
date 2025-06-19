#pragma once

#include "../types/matrix.h"
#include "is_complex.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <type_traits>

namespace LinAlgTools::Core {

class RandomGenerator {
        using Index = Core::Indices::Index;

        template<typename T>
        using BaseType = LinAlgTools::Core::UnderlyingType<T>;

public:
        explicit RandomGenerator(int32_t seed) : rng_(seed) {};

        template<typename T>
        T GetRandomTypeValue(BaseType<T> min_value,
                             BaseType<T> max_value) {
                using DistributionType = typename std::conditional<
                        std::is_integral_v<BaseType<T>>,
                        std::conditional_t<
                                IsComplexType<T>,
                                std::uniform_int_distribution<BaseType<T>>,
                                std::uniform_int_distribution<T>>,
                        std::conditional_t<
                                IsComplexType<T>,
                                std::uniform_real_distribution<BaseType<T>>,
                                std::uniform_real_distribution<T>>>::type;

                DistributionType distribution(min_value, max_value);
                if constexpr (Core::IsComplexType<T>) {
                        BaseType<T> real = distribution(rng_);
                        BaseType<T> imaginary = distribution(rng_);
                        return T{real, imaginary};
                }
                return distribution(rng_);
        }

        template<typename T>
        Matrix<T> GetRandomDenseMatrix(Index rows, Index columns,
                                       BaseType<T> min_value,
                                       BaseType<T> max_value) {
                return GetRandomSparseMatrix<T>(rows, columns, min_value, max_value, 1);
        }

        template<typename T>
        Matrix<T> GetRandomSparseMatrix(Index rows, Index columns,
                                        BaseType<T> min_value,
                                        BaseType<T> max_value,
                                        double density = 0.1) {
                assert(density >= 0.0 && density <= 1.0 &&
                       "Density must be between 0.0 and 1.0");
                Matrix<T> result(rows, columns);
                if (density == 1.0) {
                        for (int32_t i = 0; i < rows; i++) {
                                for (int32_t j = 0; j < columns; j++) {
                                        result(i, j) = GetRandomTypeValue<T>(min_value, max_value);
                                }
                        }
                        return result;
                }

                Index total_elements = rows * columns;
                Index non_zero_elements = static_cast<int32_t>(total_elements * density);

                if (density > 0.0 && non_zero_elements == 0) {
                        non_zero_elements = 1;
                }

                for (int32_t i = 0; i < non_zero_elements; ++i) {
                        Index row = static_cast<Index>(GetRandomTypeValue<int>(0, rows - 1));
                        Index column = static_cast<Index>(GetRandomTypeValue<int>(0, columns - 1));
                        result(row, column) = GetRandomTypeValue<T>(min_value, max_value);
                }

                return result;
        }

private:
        std::mt19937 rng_;
};
}//namespace LinAlgTools::Core

