#pragma once

#include <cstddef>
#include <cstdint>

namespace LinAlgTools::Core {
struct Indices 
{
        using Index = int64_t;

        struct Slice
        {
                Index first = -1;
                Index last = -1;
        };
};
}// namespace LinAlgTools::Core

