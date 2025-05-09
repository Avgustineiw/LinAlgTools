
#pragma once

#include <cstddef>
#include <cstdint>

namespace LinAlgTools::Helpers {
struct Types
{
        using Index = int64_t;

        struct RowSlice
        {
                Index begin = -1;
                Index end = -1;
        };
        struct ColumnSlice
        {
                Index begin = -1;
                Index end = -1;
        };
};
}// namespace LinAlgTools::Helpers
