import sys
import dpnp as np
import time

import numba_dpex as dpex
from numba import int32
import dpctl


@dpex.kernel
def pi_kernel(x, y, partial_hits):
    gidx = dpex.get_global_id(0)
    lidx = dpex.get_local_id(0)
    gridx = dpex.get_group_id(0)
    group_size = dpex.get_local_size(0)
    
    # There is an issue where the local memory should manual fixed: https://github.com/IntelPython/numba-dpex/issues/829
    # instead of fixing it, it should use group_size
    local_hits = dpex.local.array(8192, int32)

    local_hits[lidx] = 0

    for i in range(16):
        z = x[16*gidx+i]*x[16*gidx+i]+y[16*gidx+i]*y[16*gidx+i]
        if (z <= 1.0):
            local_hits[lidx] += 1

    # Loop for computing local_sums : divide workgroup into 2 parts
    stride = group_size // 2
    while stride > 0:
        # Waiting for each 2x2 addition into given workgroup
        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # Add elements 2 by 2 between lidx and lidx + stride
        if lidx < stride:
            local_hits[lidx] += local_hits[lidx + stride]

        stride >>= 1

    if lidx == 0:
        partial_hits[gridx] = local_hits[0]
    




def calc_pi(x, y):
    ls = dpctl.select_default_device().max_work_group_size
    print(f"max_work_group_size = {ls}")

    gs = len(x) // 16
    nb_work_groups = gs // ls
    print("ngroups = {}".format(nb_work_groups))
    
    partial_hits = np.zeros(nb_work_groups, dtype=np.int32)
    pi_kernel[dpex.NdRange(dpex.Range(gs), dpex.Range(ls))](x, y, partial_hits)
    
    hits = 0
    # calculate the final hits in host
    for i in range(nb_work_groups):
        hits += partial_hits[i]
        
    my_pi = 4.0*hits/len(x)
    
    return my_pi


def main(argv):
    if len(argv)==2:
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

    np.random.seed(17) # set the random number generator seed
    x = np.random.uniform(-1.0, 1.0, N).astype(npfloat)
    y = np.random.uniform(-1.0, 1.0, N).astype(npfloat)

    t1 = time.perf_counter()
    my_pi = calc_pi(x, y)
    t2 = time.perf_counter()
    print("N = {}  pi = {}".format(N, my_pi))
    print("calc_pi took {} s.".format(t2-t1))

if __name__ == "__main__":
    main(sys.argv[1:])
