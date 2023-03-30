import sys
import dpctl
import numpy as np
import time

def bodyForce( mass, x, y, z, velx, vely, velz, G, dt, softeningSquared ):

    for i in range(len(x)):
        dx = x[i] - x
        dy = y[i] - y
        dz = z[i] - z
                
        distSqr = (dx*dx + dy*dy + dz*dz + softeningSquared)
        invDist = 1/distSqr
        invDist3 = invDist * invDist * invDist
        
        g_mass = np.multiply(G,mass)
        g_mass[i] = 0 # To invalidate itself

        ax = np.sum(g_mass * dx * invDist3)
        ay = np.sum(g_mass * dy * invDist3)
        az = np.sum(g_mass * dz * invDist3)
                
        velx[i] = dt*ax
        vely[i] = dt*ay
        velz[i] = dt*az
	
def integrate( x, y, z, velx, vely, velz, dt ):
    x = x + velx*dt
    y = y + vely*dt
    z = z + velz*dt
	
	
def solutionPos( x, y, z ):
    pos_global = np.sum(np.sqrt(x*x+y*y+z*z))	
    return(pos_global)

def main(argv):
    if len(argv)>0 and len(argv)<4:
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

    np.random.seed(17)                            # set the random number generator seed
    softSqred = npfloat(0.001)                    # softening length
    G         = npfloat(6.674e-11)                # Newton's Gravitational Constant
    mass  = 20*np.ones(N, dtype=npfloat)/N        # total mass of particles is 20
    posx  = np.random.randn(N).astype(npfloat)    # randomly selected positions and velocities
    posy  = np.random.randn(N).astype(npfloat)
    posz  = np.random.randn(N).astype(npfloat)
    velx  = np.zeros(N, dtype=npfloat)
    vely  = np.zeros(N, dtype=npfloat)
    velz  = np.zeros(N, dtype=npfloat)

    dt = npfloat(0.1)

    t0 = time.time()
    for iter in range(iters):
        # calculate gravitational accelerations
        bodyForce(mass, posx, posy, posz, velx, vely, velz, G, dt, softSqred)
        integrate(posx, posy, posz, velx, vely, velz, dt)
    
    t1 = time.time()
    totalTime = t1-t0

    print("{} Bodies with {}iterations. {:.2f} Millions Interactions/second".format(N, iters, 1e-6*iters*N*N/totalTime))
    print("nbody took {:4.2f} s.".format(totalTime))

    print('pos={}'.format(solutionPos(posx, posy, posz)))

if __name__ == "__main__":
    main(sys.argv[1:])
