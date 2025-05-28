#pragma once

#include "../types/matrix.h"
#include "is_complex.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace LinAlgTools::Core {
namespace Implementation {
template<typename T>
struct UnderlyingT
{
        using type = T;
};

template<typename T>
struct UnderlyingT<std::complex<T>>
{
        using type = T;
};

template<typename T>
using UnderlyingType = typename UnderlyingT<T>::type;
}//namespace Implementation

template<typename T>
class RandomGenerator {
        using Index = Core::Indices::Index;
        using BaseType = Implementation::UnderlyingType<T>;
        using DistributionType = typename std::conditional<
                std::is_integral_v<BaseType>,
                std::conditional_t<
                        IsComplexType<T>,
                        std::uniform_int_distribution<BaseType>,
                        std::uniform_int_distribution<T>>,
                std::conditional_t<
                        IsComplexType<T>,
                        std::uniform_real_distribution<BaseType>,
                        std::uniform_real_distribution<T>>>::type;

public:
        explicit RandomGenerator(int32_t seed): rng_(seed) {};

        T GetRandomTypeValue(BaseType from, BaseType to) {
                DistributionType distribution(from, to);
                if constexpr (Core::IsComplexType<T>) {
                        BaseType real = distribution(rng_);
                        BaseType imaginary = distribution(rng_);
                        return T{real, imaginary};
                }
                return distribution(rng_);
        }

        int32_t GetRandomInt(int32_t from, int32_t to) {
                assert(from <= to &&
                       "Invalid range");
                std::uniform_int_distribution<int32_t> dist(from, to);
                return dist(rng_);
        }

        Matrix<T> GetRandomDenseMatrix(Index row, Index columns,
                                       BaseType from, BaseType to) {
                Matrix<T> result(row, columns);
                result.Elementwise([&](T& value) {
                        value = GetRandomTypeValue(from, to);
                });
                return result;
        }

        Matrix<T> GetRandomSparseMatrix(Index rows, Index cols,
                                        BaseType from, BaseType to,
                                        double density = 0.1) {
                assert(density >= 0.0 && density <= 1.0 &&
                       "Density must be between 0.0 and 1.0");
                Matrix<T> result(rows, cols);

                Index total_elements = rows * cols;
                Index non_zero_elements = static_cast<int32_t>(total_elements * density);

                if (density > 0.0 && non_zero_elements == 0) {
                        non_zero_elements = 1;
                }

                for (int32_t i = 0; i < non_zero_elements; ++i) {
                        Index row = static_cast<Index>(GetRandomInt(0, rows - 1));
                        Index col = static_cast<Index>(GetRandomInt(0, cols - 1));
                        result(row, col) = GetRandomTypeValue(from, to);
                }

                return result;
        }

private:
        std::mt19937 rng_;
};
}//namespace LinAlgTools::Core

