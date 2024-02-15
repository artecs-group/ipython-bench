import sys
import dpctl
import dpnp as np
import time
import numba_dpex as dpex

@dpex.kernel
def bodyForce( mass, x, y, z, velx, vely, velz, G, dt, softeningSquared):

    i = dpex.get_global_id(0)
    # BUG: local variables does not work writing more than once on them, the temporal fix is using private.array
    # https://github.com/IntelPython/numba-dpex/issues/829
    a = dpex.private.array(shape=3, dtype=np.float32)
    d = dpex.private.array(shape=3, dtype=np.float32)
    dist = dpex.private.array(shape=1, dtype=np.float32)
    g_mass = dpex.private.array(shape=1, dtype=np.float32)
    a[0] = 0.0
    a[1] = 0.0
    a[2] = 0.0

    for j in range(len(x)):
        d[0] = x[i] - x[j]
        d[1] = y[i] - y[j]
        d[2] = z[i] - z[j]

        dist[0] = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2] + softeningSquared[0])
        dist[0] = 1/dist[0]
        dist[0] = dist[0] * dist[0] * dist[0]

        g_mass[0] = G[0] * mass[j]
        if i==j:
            g_mass[0] = 0.0 # To invalidate itself

        a[0] += g_mass[0] * d[0] * dist[0]
        a[1] += g_mass[0] * d[1] * dist[0]
        a[2] += g_mass[0] * d[2] * dist[0]

    velx[i] = dt[0]*a[0]
    vely[i] = dt[0]*a[1]
    velz[i] = dt[0]*a[2]

   

@dpex.kernel
def integrate( x, y, z, velx, vely, velz, dt):
    i = dpex.get_global_id(0)

    x[i] += velx[i]*dt[0]
    y[i] += vely[i]*dt[0]
    z[i] += velz[i]*dt[0]
	
	
def solutionPos( x, y, z ):
    pos_global = np.sum(np.sqrt(x*x+y*y+z*z))	
    return(pos_global)


def main(argv):
    if len(argv)==3:
        N     = int(argv[0])
        iters = int(argv[1])
        dtypefp = argv[2]
        if dtypefp=='float32':
            size_dtypefp = np.dtype(np.float32).itemsize
            npfloat = np.float32
        else: 
            size_dtypefp = np.dtype(np.float64).itemsize
            npfloat = np.float64
    else: 
        sys.exit("Not parameters were found: python nbody.py <NBodies> <Niters> <float32/64>")

    d = dpctl.select_default_device()
    d.print_device_info()

    np.random.seed(17)                            # set the random number generator seed
    softSqred = np.full(1, 0.001, dtype=npfloat)  # softening length
    G         = np.full(1, 6.674e-11, dtype=npfloat) # Newton's Gravitational Constant
    mass  = np.random.uniform(0, 20/N, N).astype(npfloat)   # total mass of particles is 20
    posx  = np.random.uniform(-1, 1, N).astype(npfloat)   # randomly selected positions and velocities
    posy  = np.random.uniform(-1, 1, N).astype(npfloat)
    posz  = np.random.uniform(-1, 1, N).astype(npfloat)
    velx  = np.random.uniform(-1, 1, N).astype(npfloat)
    vely  = np.random.uniform(-1, 1, N).astype(npfloat)
    velz  = np.random.uniform(-1, 1, N).astype(npfloat)

    dt = np.full(1, 0.1, dtype=npfloat)

    t0 = time.time()
    for iter in range(iters):
        # calculate gravitational accelerations
        bodyForce[dpex.NdRange(dpex.Range(N), dpex.DEFAULT_LOCAL_SIZE)]( mass, posx, posy, posz, velx, vely, velz, G, dt, softSqred)
        integrate[dpex.NdRange(dpex.Range(N), dpex.DEFAULT_LOCAL_SIZE)](posx, posy, posz, velx, vely, velz, dt)
    
    t1 = time.time()
    totalTime = t1-t0

    print("{} Bodies with {} iterations. {:.2f} Millions Interactions/second".format(N, iters, 1e-6*iters*N*N/totalTime))
    print("nbody took {:4.2f} s.".format(totalTime))

    print('pos={}'.format(solutionPos(posx, posy, posz)))

if __name__ == "__main__":
    main(sys.argv[1:])
