#pragma once

#include "../../../src/algorithms/qr_decomposition.h"
#include "../../../src/algorithms/schur_decomposition.h"
#include "../../../src/algorithms/svd.h"
#include "../../../src/core/is_complex.h"
#include "../../../src/core/random_generator.h"

#include <cassert>
#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>

namespace LinAlgTools::Tests::Utils {
enum class AlgorithmName
{
        HouseholderQR,
        GivensQR,
        RealSchur,
        NaiveSVD,
};

namespace Implementation {
struct TimingResult
{
        double mean = 0;
        double min = 0;
        double max = 0;
        double stddev = 0.0;
        size_t count = 0;
};

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::milliseconds;
namespace Impl = LinAlgTools::Algorithm::Implementation;

template<typename M>
using ResultVariant = std::variant<
        Impl::PairQR<typename M::ElementType>,
        Impl::PairSchur<typename M::ElementType>,
        Impl::PairHessenberg<typename M::ElementType>,
        Impl::TripletSVD<typename M::ElementType>
>;

template<typename M>
ResultVariant<M> RunAlgorithm(AlgorithmName algorithm, const M& matrix) {
    switch (algorithm) {
        case AlgorithmName::HouseholderQR:
            return LinAlgTools::Algorithm::HouseholderQR(matrix);
        case AlgorithmName::GivensQR:
            return LinAlgTools::Algorithm::GivensQR(matrix);
        case AlgorithmName::RealSchur:
            return LinAlgTools::Algorithm::RealSchur(matrix);
        case AlgorithmName::NaiveSVD:
            return LinAlgTools::Algorithm::NaiveSVD(matrix);
        default:
                assert("Unknown algorithm");
    }
}

inline TimingResult CalculateStatistics(const std::vector<double>& data) {
        TimingResult result;
        if (data.empty()) {
                return result;
        }

        result.count = data.size();

        result.min = data[0];
        result.max = data[0];
        double sum = 0;

        for (const auto t: data) {
                sum += t;
                if (t < result.min) result.min = t;
                if (t > result.max) result.max = t;
        }

        result.mean = sum / static_cast<double>(result.count);

        if (result.count > 1) {
                double variance = 0.0;
                for (const auto t: data) {
                        variance += std::pow(t - result.mean, 2);
                }
                result.stddev = std::sqrt(variance / result.count);
        }

        return result;
}

template<typename T = std::complex<long double>>
TimingResult GetTimingStatistics(AlgorithmName algorithm,
                                 Core::RandomGenerator generator,
                                 int32_t size, int32_t iterations,
                                 LinAlgTools::Core::UnderlyingType<T> min_value,
                                 LinAlgTools::Core::UnderlyingType<T> max_value) {
        std::vector<double> data(iterations);

        for (int32_t i = 0; i < iterations; i++) {
                auto matrix = generator.GetRandomDenseMatrix<T>(size, size, min_value, max_value);

                Clock::time_point start = Clock::now();
                auto result = RunAlgorithm(algorithm, matrix);
                Clock::time_point end = Clock::now();

                data[i] = std::chrono::duration_cast<Ms>(end - start).count();
        }

        return CalculateStatistics(data);
}

inline std::string GetNameOfAlgorithm(AlgorithmName algorithm) {
        switch (algorithm) {
                case AlgorithmName::HouseholderQR:
                        return "HouseholderQR";
                        break;
                case AlgorithmName::GivensQR:
                        return "GivensQR";
                        break;
                case AlgorithmName::RealSchur:
                        return "RealSchur";
                        break;
                case AlgorithmName::NaiveSVD:
                        return "NaiveSVD";
                        break;
                default:
                        assert("Unknown algorithm");
        }
}
}//namespace Implementation

template<typename T = std::complex<long double>>
void RunPerformanceTest(AlgorithmName algorithm,
                        int32_t min_size = 1, int32_t max_size = 100,
                        int32_t size_step = 1, int32_t matrices_per_iteration = 10,
                        Core::RandomGenerator generator = Core::RandomGenerator(20),
                        LinAlgTools::Core::UnderlyingType<T> min_value = 1e-10,
                        LinAlgTools::Core::UnderlyingType<T> max_value = 1e+10) {
        std::string algorithm_name = Implementation::GetNameOfAlgorithm(algorithm);

        std::filesystem::path filename = algorithm_name + "_performance.csv";
        std::ofstream outFile(filename, std::ios::app);

        std::cout << "\nPerformance Test (" << algorithm_name << ")\n";
        std::cout << "---------------------------------\n";
        std::cout << std::setw(10) << "Size"
                  << std::setw(15) << "Mean (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)"
                  << std::setw(15) << "StdDev\n";

        outFile << "Algorithm,Size,Mean(ms),Min(ms),Max (ms),StdDev\n";

        for (int32_t size = min_size; size <= max_size; size += size_step) {
                auto stats = Implementation::GetTimingStatistics(algorithm, generator,
                                                                 size, matrices_per_iteration,
                                                                 min_value, max_value);

                std::cout << std::setw(10) << size
                          << std::setw(15) << stats.mean
                          << std::setw(15) << stats.min
                          << std::setw(15) << stats.max
                          << std::setw(15) << std::fixed << std::setprecision(2)
                          << stats.stddev << '\n';

                outFile << algorithm_name << ","
                        << size << ","
                        << stats.mean << ","
                        << stats.min << ","
                        << stats.max << ","
                        << std::fixed << std::setprecision(6)
                        << stats.stddev << '\n';
        }

        std::cout << "\nResults saved to: " << filename << '\n';
}
}//namespace LinAlgTools::Tests::Utils

