#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include <iostream>

using namespace cl::sycl;


float rand_uniform()
{
    return float(rand()) / RAND_MAX;
}

bool verify_result(int m, int n, int k, int ldc, float *C, float *C_reference)
{
    float tolerance = 1e-2;
    bool ok = true;

    // Compare host side results with the result buffer from device side: print
    // fail data 5 times only.
    int printf_count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto idx = i * ldc + j;
            auto abs_diff = std::abs(C[idx] - C_reference[idx]);

            if (abs_diff > tolerance && printf_count++ < 5) {
                std::cerr << "The result is incorrect for element "
                          << '[' << i << ", " << j << ']'
                          << ", expected: " << C_reference[idx]
                          << ", but got: " << C[idx] << std::endl;
                ok = false;
            }
        }
    }

    if (ok)
        std::cout << "Results are accurate.\n";

    return ok;
}

int main(int argc, char* argv[]) {
        // Matrix data sizes.
        //
        // A is m x k
        // B is k x n  --> product C is m x n

        if (argc != 4) {
                std::cout << "Parameters are not correct." << std::endl
                << "./main <m> <k> <n> #Cmn=Amk·Bkn" << std::endl;
                exit(-1);
        }

        int m =  atoi(argv[1]);
        int k =  atoi(argv[2]);
        int n =  atoi(argv[3]);

        // Leading dimensions of data. For row-major matrices, the leading
        // dimension is the stride between adjacent rows.
        int lda = k;
        int ldb = n;
        int ldc = n;

        // Scaling factors.
        float alpha = 1.0f;
        float beta = 0.0f;

        // Create a queue on the default device.
        std::uint64_t seed{0};
        sycl::queue device_queue{sycl::default_selector_v};

        std::cout << std::endl << "Running on: "
                << device_queue.get_device().get_info<sycl::info::device::name>()
                << std::endl << std::endl;


        // Allocate shared memory for matrices.
        auto transA = oneapi::mkl::transpose::nontrans;
        auto transB = oneapi::mkl::transpose::nontrans;
        auto A = malloc_shared<float>(m * k, device_queue);
        auto B = malloc_shared<float>(k * n, device_queue);
        auto C = malloc_shared<float>(m * n, device_queue);
        auto C_reference = (float *) calloc(m * n, sizeof(float));

        if (!A || !B || !C || !C_reference) {
                std::cerr << "Could not allocate memory for matrices." << std::endl;
                exit(1);
        }

        // Initialize matrix data.
        for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                    A[i * lda + j] = rand_uniform();

        for (int i = 0; i < k; i++)
                for (int j = 0; j < n; j++)
                    B[i * ldb + j] = rand_uniform();

        std::cout << "Problem size: "
                      << " A (" << m << 'x' << k << ") *"
                      << " B (" << k << 'x' << n << ")  --> "
                      << " C (" << m << 'x' << n << ")\n";

        // Call GEMM to do matrix multiplication, asynchronously.
        auto begin = std::chrono::high_resolution_clock::now(); // Start measuring time
        std::cerr << "Launching oneMKL GEMM calculation..." << std::endl;
        oneapi::mkl::blas::row_major::gemm(device_queue, transA, transB, m, n, k,
                                               alpha, A, lda, B, ldb, beta, C, ldc);

	device_queue.wait();
        auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time and calculate the elapsed time
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        std::cout << std::endl << "GEMM took = " << elapsed.count() * 1e-9 << " (s)" << std::endl << std::endl;

        std::cout << "Global Matrices sizes="
                << (m*k+k*n+m*n)*sizeof(float)/1024/1024
                << "MB"
                << std::endl;

        // While calculation occurs, compute reference result to check accuracy.
        // Compute C = A * B on CPU
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                alpha, A, lda, B, ldb, beta, C_reference, ldc);

        // Check results for accuracy.
        bool ok = verify_result(m, n, k, ldc, C, C_reference);

        // Free memory.
        free(A, device_queue);
        free(B, device_queue);
        free(C, device_queue);
        free(C_reference);
}
