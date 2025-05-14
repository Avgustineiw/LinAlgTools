#pragma once

#include "../../src/types/matrix.h"

#include <cassert>
#include <cstdint>
#include <random>

namespace LinAlgTools::Tests {
template<typename T>
class RandomGenerator {
        using Index = Core::Indices::Index;

public:
        RandomGenerator(int32_t seed, int32_t minValue = INT32_MIN, int32_t maxValue = INT32_MAX)
            : rng_(seed),
              minValue_(minValue), maxValue_(maxValue),
              distribution_(minValue, maxValue) {
                assert(minValue >= INT32_MIN && minValue_ <= INT32_MAX &&
                       "Incorrect minimum value overflow.");
                assert(maxValue >= INT32_MIN && maxValue_ <= INT32_MAX &&
                       "Incorrect maximum value overflow.");
        };

        T GetRandomTypeValue() {
                if constexpr (Core::IsComplexType<T>) {
                        using ValueType = typename T::value_type;
                        std::uniform_real_distribution<ValueType> real_dist(minValue_, maxValue_);

                        ValueType real = real_dist(rng_);
                        ValueType imaginary = real_dist(rng_);
                        return T{real, imaginary};
                }
                return static_cast<T>(distribution_(rng_));
        }

        Matrix<T> GetRandomMatrix(Index size) {
                return GetRandomMatrix(size, size);
        }

        Matrix<T> GetRandomMatrix(Index row, Index columns) {
                Matrix<T> result(row, columns);
                result.Elementwise([&](T& value) {
                        value = GetRandomTypeValue();
                });
                return result;
        }

private:
        std::mt19937 rng_;
        const int32_t minValue_;
        const int32_t maxValue_;
        std::uniform_int_distribution<int32_t> distribution_;
};
}// namespace LinAlgTools::Tests
