import sys
import numpy as np
import time

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

    x = np.random.uniform(-1.0, 1.0, N).astype(npfloat)
    y = np.random.uniform(-1.0, 1.0, N).astype(npfloat)

    t1 = time.perf_counter()
    z = np.multiply(x, x) + np.multiply(y,y)
    hits = np.count_nonzero(z <=1.0)
    my_pi = 4.0*hits/N
    t2 = time.perf_counter()
    print("N={}  pi={}".format(N, my_pi))
    print("calc_pi took {} s.".format(t2-t1))


if __name__ == "__main__":
    main(sys.argv[1:])
