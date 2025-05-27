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
        using DistributionType = typename std::conditional<
                std::is_integral_v<Implementation::UnderlyingType<T>>,
                std::conditional_t<
                        IsComplexType<T>,
                        std::uniform_int_distribution<Implementation::UnderlyingType<T>>,
                        std::uniform_int_distribution<T>>,
                std::conditional_t<
                        IsComplexType<T>,
                        std::uniform_real_distribution<Implementation::UnderlyingType<T>>,
                        std::uniform_real_distribution<T>>>::type;

public:
        RandomGenerator(int32_t seed,
                        int32_t minValue = INT32_MIN,
                        int32_t maxValue = INT32_MAX)
            : rng_(seed),
              minValue_(minValue), maxValue_(maxValue),
              distribution_(minValue, maxValue) {
                assert(minValue >= INT32_MIN &&
                       minValue <= INT32_MAX &&
                       "Invalid minimum value");
                assert(maxValue >= INT32_MIN &&
                       maxValue <= INT32_MAX &&
                       "Invalid maximum value");
                assert(minValue <= maxValue &&
                       "Invalid range");
        }

        T GetRandomTypeValue() {
                if constexpr (Core::IsComplexType<T>) {
                        using ValueType = typename T::value_type;
                        ValueType real = distribution_(rng_);
                        ValueType imaginary = distribution_(rng_);
                        return T{real, imaginary};
                }
                return distribution_(rng_);
        }

        int32_t GetRandomInt(int32_t from, int32_t to) {
                assert(from <= to &&
                       "Invalid range");
                std::uniform_int_distribution<int32_t> dist(from, to);
                return dist(rng_);
        }

        Matrix<T> GetRandomDenseMatrix(Index size) {
                return GetRandomDenseMatrix(size, size);
        }

        Matrix<T> GetRandomDenseMatrix(Index row, Index columns) {
                Matrix<T> result(row, columns);
                result.Elementwise([&](T& value) {
                        value = GetRandomTypeValue();
                });
                return result;
        }

        Matrix<T> GetRandomSparseMatrix(Index size, double density = 0.1) {
                return GetRandomSparseMatrix(size, size, density);
        }

        Matrix<T> GetRandomSparseMatrix(Index rows, Index cols, double density = 0.1) {
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
                        result(row, col) = GetRandomTypeValue();
                }

                return result;
        }

private:
        std::mt19937 rng_;
        const int32_t minValue_;
        const int32_t maxValue_;
        DistributionType distribution_;
};
}//namespace LinAlgTools::Core

