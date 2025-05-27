#pragma once

#include "../types/matrix.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace LinAlgTools::Core {
namespace Implementation {
template<typename T>
constexpr T DefaultMin() {
        if constexpr (std::is_integral_v<T>) {
                return -100;
        }
        else {
                return T{-1e-10};
        }
}

template<typename T>
constexpr T DefaultMax() {
        if constexpr (std::is_integral_v<T>) {
                return 100;
        }
        else {
                return T{1e+20};
        }
}
}//namespace implementation

template<typename T>
class RandomGenerator {
        using Index = Core::Indices::Index;
        using DistributionType = typename std::conditional<
                std::is_integral_v<T>,
                std::uniform_int_distribution<T>,
                std::uniform_real_distribution<T>>::type;

public:
        RandomGenerator(int32_t seed,
                        T minValue = Implementation::DefaultMin<T>(), 
                        T maxValue = Implementation::DefaultMax<T>()) 
            : rng_(seed),
              minValue_(minValue),
              maxValue_(maxValue) {
                assert(minValue <= maxValue &&
                       "Invalid range");

                if constexpr (std::is_integral_v<T>) {
                        distribution_ = std::uniform_int_distribution<T>(minValue, maxValue);
                }
                else if constexpr (std::is_floating_point_v<T>) {
                        distribution_ = std::uniform_real_distribution<T>(minValue, maxValue);
                }
        }

        T GetRandomTypeValue() {
                if constexpr (Core::IsComplexType<T>) {
                        using ValueType = typename T::value_type;
                        std::uniform_real_distribution<ValueType> real_dist(minValue_, maxValue_);

                        ValueType real = real_dist(rng_);
                        ValueType imaginary = real_dist(rng_);
                        return T{real, imaginary};
                }
                else if constexpr (std::is_integral_v<T>) {
                        return distribution_(rng_);
                }
                else if constexpr (std::is_floating_point_v<T>) {
                        return distribution_(rng_);
                }
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
