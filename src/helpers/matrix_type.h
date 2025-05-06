#pragma once

#include <type_traits>

namespace LinAlgTools {
template <typename T> class Matrix;
template <typename T> class SubMatrix;
template <typename T> class ConstSubMatrix;
} // namespace LinAlgTools

namespace LinAlgTools::Helpers {
template <typename T> 
struct IsMatrixT : std::false_type {};

template <typename T>
struct IsMatrixT<Matrix<T>> : std::true_type {};

template <typename T>
struct IsMatrixT<SubMatrix<T>> : std::true_type {};

template <typename T>
struct IsMatrixT<ConstSubMatrix<T>> : std::true_type {};

template <typename T>
struct IsMutableMatrixT : std::false_type {};

template <typename T>
struct IsMutableMatrixT<Matrix<T>> : std::true_type {};

template <typename T>
struct IsMutableMatrixT<SubMatrix<T>> : std::true_type {};


template <typename T>
concept MatrixType = IsMatrixT<std::remove_cv_t<T>>::value;

template <typename T>
concept MutableMatrixType = IsMutableMatrixT<std::remove_cv_t<T>>::value;
} // namespace Helpers
