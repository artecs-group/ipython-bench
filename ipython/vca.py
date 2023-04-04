# -*- coding: utf-8 -*-
import sys
import os
import struct
import time
import math
from typing import Tuple
import dpnp as np
import dpctl

#############################################
# Internal functions
#############################################

def pinv(A: np.ndarray, dtype) -> np.ndarray:
    rcond = np.asarray(1.1920929e-07, dtype=dtype)
    u, s, vt = np.linalg.svd(A, full_matrices=False)

    # discard small singular values
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1)
    large = s > cutoff
    s = np.divide(1, s, where=large, out=s, dtype=dtype)
    s = np.where(large, s, 0)

    res = np.matmul(vt.T, np.multiply(s[..., np.newaxis], u.T))
    return res


def estimate_snr(Y: np.ndarray, r_m: np.ndarray, x: np.ndarray) -> float:
    [L, N] = Y.shape           # L number of bands (channels), N number of pixels
    [p, N] = x.shape           # p number of endmembers (reduced dimension)

    P_y     = np.sum(np.square(Y)) / float(N)
    P_x     = np.sum(np.square(x)) / float(N) + np.sum(np.square(r_m))
    snr_est = 10 * math.log10((P_x - p / L * P_y) / (P_y - P_x))
    return snr_est



def vca(Y: np.ndarray, R: int, verbose: bool = False, snr_input: int = 0, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Vertex Component Analysis
    #
    # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - matrix with dimensions L(channels) x N(pixels)
    #      each pixel is a linear mixture of R endmembers
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.   
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------
    #
    # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
    # This code is a translation of a matlab code provided by 
    # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
    #
    # more details on:
    # Jose M. P. Nascimento and Jose M. B. Dias 
    # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
    # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
    # 
    # 

    #############################################
    # Initializations
    #############################################
    if len(Y.shape) != 2:
        sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N] = Y.shape   # L number of bands (channels), N number of pixels
        
    R = int(R)
    if (R < 0 or R > L):  
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')
        
    #############################################
    # SNR Estimates
    #############################################

    if snr_input == 0:
        y_m = np.mean(Y, axis=1)
        Y_o = (Y.T - y_m).transpose()           # data with zero-mean
        #splin.lapack.dgesvd
        Ud  = np.linalg.svd(np.divide(np.matmul(Y_o, Y_o.T), N, dtype=dtype))[0][:,:R]  # computes the R-projection matrix 
        x_p = np.matmul(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * math.log10(R)
            
    #############################################
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")
                
        d = R-1
        if snr_input  == 0: # it means that the projection is already computed
            Ud = Ud[:,:d]
        else:
            y_m = np.mean(Y, axis=1)
            Y_o = (Y.T - y_m).transpose()  # data with zero-mean 
                
            Ud  = np.linalg.svd(np.divide(np.matmul(Y_o, Y_o.T), N, dtype=dtype))[0][:,:d]  # computes the p-projection matrix 
            x_p =  np.matmul(Ud.T, Y_o)                 # project the zeros mean data onto p-subspace

            Yp =  (np.matmul(Ud, x_p[:d,:]).transpose() + y_m).transpose()     # again in dimension L
                    
            x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = math.sqrt(np.amax(np.sum(np.square(x), axis=1, dtype=dtype)))
            y = np.vstack(( x, c * np.ones((1, N), dtype=dtype)))
    else:
        if verbose:
            print("... Select the projective proj.")
                
        d = R
        Ud  = np.linalg.svd(np.divide(np.matmul(Y, Y.T), N, dtype=dtype))[0][:,:d] # computes the p-projection matrix 
        x_p = np.matmul(Ud.T, Y)[:d,:]
        Yp = np.matmul(Ud, x_p)      # again in dimension L (note that x_p has no null mean)
        x = np.matmul(Ud.T, Y)
        u = np.mean(x, axis=1)        #equivalent to  u = Ud.T * r_m
        y = x / np.matmul(u.T, x)


    #############################################
    # VCA algorithm
    #############################################

    indices = np.zeros((R), dtype=int)
    A = np.zeros((R, R), dtype=dtype)
    A[-1,0] = 1

    for i in range(R):
        w = np.random.rand(R, 1).astype(dtype)
        f = w - np.matmul(np.matmul(A, pinv(A, dtype)), w)
        f = np.divide(f, np.linalg.norm(f), dtype=dtype)

        v = np.matmul(f.T, y)

        indices[i] = np.argmax(np.absolute(v))
        #A[:,i] = y[:,indices[i]]        # same as x(:,indices(i))

    #Ae = Yp[:, indices]
    Ae = np.take(Yp, indices, axis=1)
    return Ae, indices, Yp


if __name__ == '__main__':
    if not len(sys.argv) == 5:
        sys.exit("Not parameters were found: python vca.py <path to image file, e.g.: Cuprite> <target endmembers, e.g.: 19> <SNR, e.g.: 0> <float32/64>")

    path:str = sys.argv[1]
    samples:int = 0
    lines:int = 0
    bands:int = 0
    target:int = int(sys.argv[2])
    snr:float = float(sys.argv[3])
    dtypefp:str = sys.argv[4]
    if dtypefp=='float32':
        npfloat = np.float32
    else:
        npfloat = np.float64

    print(f"Reading {path}.hdr file...")
    with open(path + ".hdr", 'r') as f:
        for line in f:
            if not line.find("samples") == -1:
                samples = int(line.split('= ')[1])
            elif not line.find("lines") == -1:
                lines = int(line.split('= ')[1])
            elif not line.find("bands") == -1:
                bands = int(line.split('= ')[1])

    print("Done.")
    print(f"Reading {path} file...")
    with open(path, 'rb') as f:
        # Read the binary data as bytes
        binary_data = f.read()
        # Use the struct module to unpack the binary data into a tuple
        float_data = struct.unpack("<" + "h" * (len(binary_data) // 2), binary_data)
        # Store the data in a list
        cup_vec = [float(i) for i in float_data]

    cup = np.asarray(cup_vec, dtype=npfloat).reshape(bands, samples*lines)
    print("Done.")
    d = dpctl.select_default_device()
    d.print_device_info()
    print("Starting VCA algorithm...")
    start = time.time()
    Ae, indices, Yp = vca(cup, target, verbose=True, snr_input=snr, dtype=npfloat)
    end = time.time()
    print(f"VCA took {end-start}s")

    print("Writing results...")
    with open('data/End-Cupriteb_python-02.txt', 'w') as f:
        for i in range(target):
            f.write(f"==={i}")
            for j in range(bands):
                f.write(f"{Ae[j, i]}")
    
    # with open('data/python_endmem.bin', 'wb') as f:
    #     f.write(Ae.tobytes())
    print("Done.")
            