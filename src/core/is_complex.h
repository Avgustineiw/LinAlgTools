#pragma once

#include <complex>
#include <type_traits>

namespace LinAlgTools::Core {
namespace Implementation {
template<typename T>
struct IsComplexType : std::false_type{};

template<typename T>
struct IsComplexType<std::complex<T>> : std::true_type{};
}//namespace LinAlgTools::Core::Implementation

template<typename T>
constexpr bool IsComplexType = Implementation::IsComplexType<std::remove_cv_t<T>>::value;
}// namespace LinAlgTools::Core
