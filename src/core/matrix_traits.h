#pragma once

#include <type_traits>

namespace LinAlgTools {
template <typename T> class Matrix;
template <typename T> class SubMatrix;
template <typename T> class ConstSubMatrix;
} // namespace LinAlgTools

namespace LinAlgTools::Core {
namespace Implementation{
template <typename T> 
struct IsMatrixType : std::false_type {};

template <typename T>
struct IsMatrixType<Matrix<T>> : std::true_type {};

template <typename T>
struct IsMatrixType<SubMatrix<T>> : std::true_type {};

template <typename T>
struct IsMatrixType<ConstSubMatrix<T>> : std::true_type {};

template <typename T>
struct IsMutableMatrixType : std::false_type {};

template <typename T>
struct IsMutableMatrixType<Matrix<T>> : std::true_type {};

template <typename T>
struct IsMutableMatrixType<SubMatrix<T>> : std::true_type {};
} //namespace LinAlgTools::Core::Implementation

template <typename T>
concept MatrixType = Implementation::IsMatrixType<std::remove_cv_t<T>>::value;

template <typename T>
concept MutableMatrixType = Implementation::IsMutableMatrixType<std::remove_cv_t<T>>::value;
} // namespace LinAlgTools::Core 

