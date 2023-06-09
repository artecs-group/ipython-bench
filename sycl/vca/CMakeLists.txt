cmake_minimum_required(VERSION 3.13)

set(This vca.exe)
set(ThisName vca)

set(Sources 
    vca.cpp
)

add_executable(${This} ${Sources})
set(CFLAGS "-O3" "-fsycl-unnamed-lambda" "-fsycl" "-qmkl" "-DMKL_ILP64")
target_compile_options(${This} PRIVATE ${CFLAGS})
target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64 $ENV{TBBROOT}/lib/intel64/gcc4.8)
target_link_libraries(${This} ${CFLAGS} "-fsycl-device-code-split=per_kernel" mkl_sycl "-Wl,-export-dynamic -Wl,--start-group" mkl_intel_ilp64 mkl_tbb_thread mkl_core "-Wl,--end-group" tbb sycl OpenCL pthread m dl)

add_custom_target(run_${ThisName}
    COMMAND ${PROJECT_BINARY_DIR}/${ThisName}/${This} ../../../data/Cuprite 19 1
    DEPENDS ${PROJECT_BINARY_DIR}/${ThisName}/${This}
    WORKING_DIRECTORY .
)