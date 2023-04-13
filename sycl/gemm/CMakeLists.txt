cmake_minimum_required(VERSION 3.13)

set(This gemm.exe)
set(ThisName gemm)

set(Sources 
    gemm.cpp
)

add_executable(${This} ${Sources})
set(CFLAGS "-O3" "-fsycl-unnamed-lambda" "-fsycl" "-qmkl" "-DMKL_ILP64")
target_compile_options(${This} PRIVATE ${CFLAGS})
target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64)
target_link_libraries(${This} ${CFLAGS} "-fsycl-device-code-split=per_kernel" mkl_sycl "-Wl,-export-dynamic -Wl,--start-group" mkl_intel_lp64 mkl_intel_thread mkl_core "-Wl,--end-group" iomp5 pthread m dl)

add_custom_target(run_${ThisName}
    COMMAND ${PROJECT_BINARY_DIR}/${ThisName}/${This} 4096 2048 1024
    DEPENDS ${PROJECT_BINARY_DIR}/${ThisName}/${This}
    WORKING_DIRECTORY .
)
