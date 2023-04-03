cmake_minimum_required(VERSION 3.13)

set(This sycl_bench)

project(${This} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_COMPILER icpx)

add_subdirectory(calc_pi)
add_subdirectory(gemm)
add_subdirectory(nbody)
add_subdirectory(vca)