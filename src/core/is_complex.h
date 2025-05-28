#pragma once

#include <complex>
#include <type_traits>

namespace LinAlgTools::Core {
namespace Implementation {
template<typename T>
struct IsComplexType : std::false_type{};

template<typename T>
struct IsComplexType<std::complex<T>> : std::true_type{};

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
}//namespace LinAlgTools::Core::Implementation

template<typename T>
inline constexpr bool IsComplexType = Implementation::IsComplexType<std::remove_cv_t<T>>::value;

template<typename T>
using UnderlyingType = typename Implementation::UnderlyingT<T>::type;
}// namespace LinAlgTools::Core

