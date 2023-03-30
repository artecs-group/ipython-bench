#Run: SYCL_DEVICE_FILTER=cpu python3 calc_pi_dpnp.py 100000000 float64
import sys
import dpnp as dp
import numpy as np
import time

import numba_dpex as dpex
import dpctl


@dpex.kernel
def vector_add_dpex(x, y):
    gidx = dpex.get_global_id(0)
    lidx = dpex.get_local_id(0);

    elements_per_work_item = 256;

    z = x[gidx]*x[gidx]+y[gidx]*y[gidx]

def main(argv):
    if len(argv)>0 and len(argv)<3:
        N = int(argv[0])
        dtypefp = argv[1]
        if dtypefp=='float32':
            size_dtypefp = np.dtype(np.float32).itemsize
            npfloat = np.float32
        else: 
            size_dtypefp = np.dtype(np.float64).itemsize
            npfloat = np.float64
    else: 
        sys.exit("Not parameters were found: python calc_pi.py <N> <float32/64>")
                
    d = dpctl.select_default_device()
    d.print_device_info()

    x = dp.random.uniform(-1.0, 1.0, N).astype(npfloat)
    y = dp.random.uniform(-1.0, 1.0, N).astype(npfloat)

    t1 = time.perf_counter()
    z = dp.multiply(x, x) + dp.multiply(y,y)
    hits = dp.count_nonzero(z <=1.0)
    my_pi = 4.0*hits/N
    t2 = time.perf_counter()
    print("N={}  pi={}".format(N, my_pi))
    print("calc_pi took {} s.".format(t2-t1))


if __name__ == "__main__":
    main(sys.argv[1:])
