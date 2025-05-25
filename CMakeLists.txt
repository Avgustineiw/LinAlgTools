cmake_minimum_required(VERSION 3.26)
project(LinAlgTools VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-p=${CMAKE_CURRENT_SOURCE_DIR}")

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

set(VALIDATION_TEST_DIR "tests/validation_tests")
set(BENCHMARK_TEST_DIR "tests/benchmark_tests")

file(GLOB VALIDATION_TESTS "${VALIDATION_TEST_DIR}/*.cpp")
file(GLOB BENCHMARK_TESTS "${BENCHMARK_TEST_DIR}/*.cpp")

if(VALIDATION_TESTS)
    add_executable(test_validation ${VALIDATION_TESTS})
    target_link_libraries(test_validation PRIVATE gtest_main)
    gtest_discover_tests(test_validation)
endif()

if(BENCHMARK_TESTS)
    add_executable(test_benchmark ${BENCHMARK_TESTS})
    target_link_libraries(test_benchmark PRIVATE gtest_main)
    gtest_discover_tests(test_benchmark)
endif()

if(VALIDATION_TESTS OR BENCHMARK_TESTS)
    add_executable(test_all ${VALIDATION_TESTS} ${BENCHMARK_TESTS})
    target_link_libraries(test_all PRIVATE gtest_main)
    gtest_discover_tests(test_all)
endif()

add_executable(main main.cpp)

add_custom_target(run_tests DEPENDS test_all)
add_custom_target(run_validation DEPENDS test_validation)
add_custom_target(run_benchmark DEPENDS test_benchmark)
