#pragma once

namespace LinAlgTools::Helpers {
template<typename T>
int sgn(T val) {
        return (T(0) < val) - (val < T(0));
}
}// namespace LinAlgTools::Helpers
