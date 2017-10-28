# Numerical schemes for simulating diffusion for outer code diffusion.py

from __future__ import absolute_import, division, print_function
import numpy as np

# The linear algebra package for BTCS (for solving the matrix equation)
import numpy.linalg as la

def diffusion_ftcs(phi, Dx, Dt, K):
    """Gives the result of diffusion of $phi after 1 timestep.
    Uses finite differences, second order, forward in time, centered in space.
    Assumes that $phi is constant at its boundaries.
    :param phi: array to diffuse
    :param Dx: space step
    :param Dt: time step
    :param K: diffusion coefficient
    :return: diffused array of same size as $phi
    """
    phiNew = []
    for i in range(len(phi)):
        if i == 0:
            #cheating
            phiNew.append(phi[i])
        elif i == len(phi) - 1:
            #cheating
            phiNew.append(phi[i])
        else:
            #forward time, centered space
            phiNew.append(phi[i] + (phi[i+1]+phi[i-1]-2*phi[i])*(K*Dt/(Dx**2)))
    return np.array(phiNew)

def diffusion_btcs(phi, Dx, Dt, K):
    """Gives the result of diffusion of $phi after 1 timestep.
    Uses finite differences, second order, backward in time, centered in space.
    :param phi: array to diffuse
    :param Dx: space step
    :param Dt: time step
    :param K: diffusion coefficient
    :return: diffused array of same size as $phi
    """
    d_coeff = K*Dt/(Dx**2)
    N = len(phi)
    #construct the thingy matrix
    M = np.eye(N) * (1 + 2*d_coeff)
    M += np.eye(N, k=-1) * (-d_coeff)
    M += np.eye(N, k=N-1) * (-d_coeff)
    M += np.eye(N, k=1) * (-d_coeff)
    M += np.eye(N, k=1-N) * (-d_coeff)
    #and solve M*phiNew = phi
    phiNew = la.solve(a=M, b=phi)
    return phiNew
