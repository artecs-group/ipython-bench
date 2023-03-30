#Run: SYCL_DEVICE_FILTER=cpu python3 matrix_mult_dpnp.py 8192 8192 8192 float32
import sys
import dpctl
import dpnp as dp
import numpy as np
import time

def main(argv):
    if len(argv)>0 and len(argv)<5:
        N = int(argv[0])
        M = int(argv[1])
        K = int(argv[2])
        dtypefp = argv[3]
        if dtypefp=='float32':
            size_dtypefp = np.dtype(np.float32).itemsize
            npfloat = np.float32
        else: 
            size_dtypefp = np.dtype(np.float64).itemsize
            npfloat = np.float64
    else: 
        sys.exit("Not parameters were found: python gemm.py <N> <M> <K> <float32/64> # Cnm=Ank·Bkm")
        
    d = dpctl.select_default_device()
    d.print_device_info()

    a = dp.ones((N,K), dtype=dtypefp)
    b = dp.random.random((K,M)).astype(npfloat)

    t1 = time.perf_counter()
    c_ref = dp.matmul(a, b)
    t2 = time.perf_counter()
    print("N={} M={} K={}".format(N, M, K))
    print("Gemm took {} s.".format(t2-t1))

    print("Global Matrices sizes={} MB".format((M*K+K*N+M*N)*size_dtypefp/1024/1024))


if __name__ == "__main__":
    main(sys.argv[1:])
