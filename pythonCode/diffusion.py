#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

#niceifying the plots with a larger font size
from matplotlib import rc
rc('font', size=12)

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
execfile("diffusionSchemes.py")
execfile("diagnostics.py")
execfile("initialConditions.py")

#initial wave defined by squareWave(x,x_min,x_max) from initialConditions.py

def diffusion(differ,N_t,Dt,N_x,x_min,x_max,K,alpha,beta):
    """Diffuses a square wave using a diffusion scheme $differ.
    :param differ: function defining diffusion scheme to use
    :param N_t: number of points in time
    :param Dt: timestep
    :param N_x: number of points in space
    :param x_min: minimum space value
    :param x_max: maximum space value
    :param K: diffusion coefficient
    :param alpha: minimum space value of square wave
    :param beta: maximum space value of square wave
    """
    #define space
    space = np.linspace(x_min,x_max,N_x)
    Dx = (x_max - x_min) / (N_x - 1)
    #define time
    time = [t * Dt for t in range(N_t)]
    #produce initial state
    phis = [squareWave(space,alpha,beta)]
    #iterate diffusion
    for _ in range(N_t - 1):
        phis.append(differ(phis[-1],Dx,Dt,K))
    return time, np.array(phis)

if __name__ == "__main__":
    #example parameters
    N_t = 40
    Dt = 0.1
    N_x = 41
    x_min = 0.0
    x_max = 1.0
    K = 1e-3
    alpha = 0.4
    beta = 0.6
    #produce diffusionses
    ftime, diff_forw = diffusion(diffusion_ftcs,N_t,Dt,N_x,x_min,x_max,K,alpha,beta)
    btime, diff_back = diffusion(diffusion_btcs,N_t,Dt,N_x,x_min,x_max,K,alpha,beta)
    #calculate error at EACH timestep
    space = np.linspace(x_min,x_max,N_x)
    diff_actual = np.array([[analyticErf(x,K*Dt*t_i,alpha,beta) for x in space] for t_i in range(N_t)])
    errors_forw = []
    errors_back = []
    for i in range(0,len(diff_actual)):
        errors_forw.append(L2ErrorNorm(diff_forw[i],diff_actual[i]))
        errors_back.append(L2ErrorNorm(diff_back[i],diff_actual[i]))
        ##print("t:{0}\tforw:{1:.3e}\tback:{2:.3e}".format(i*Dt,errors_forw[-1],errors_back[-1]))
    #plot results
    plt.suptitle("Diffusion of square wave by finite differences, second order")
    plt.subplot(221)
    plt.plot(space, diff_forw[-1])
    plt.plot(space, diff_actual[-1])
    plt.title("Forward in time, centered in space")
    plt.xlabel("Space")
    plt.ylabel("Concentration")
    plt.subplot(222)
    plt.plot(space, diff_back[-1])
    plt.plot(space, diff_actual[-1])
    plt.title("Backward in time, centered in space")
    plt.xlabel("Space")
    plt.ylabel("Concentration")
    plt.subplot(223)
    plt.plot(ftime, errors_forw)
    plt.title("Root mean square error (FTCS)")
    plt.xlabel("Time")
    plt.ylabel("RMS error")
    plt.subplot(224)
    plt.plot(btime, errors_back)
    plt.title("Root mean square error (BTCS)")
    plt.xlabel("Time")
    plt.ylabel("RMS error")
    plt.subplots_adjust(left=0.07,bottom=0.1,right=0.97,top=0.9,wspace=0.17,hspace=0.4)
    plt.show(block=False)
