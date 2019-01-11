import sys
import math
import numpy as np
from multiprocessing import pool
from numpy import linalg as LA

"""
these are symmetry functions used as
input to the neural network
"""

def cutoff_function( rmag , Rc ):
    """
    this is cutoff function for
    Gaussian symmetry functions
    """
    if ( rmag <= Rc ):
        fc = 0.5 * ( np.cos( np.pi * rmag / Rc ) + 1 )
    else:
        fc = 0.0

    return fc


def radial_gaussian(rij, i_atom , width, rshift, Rc, Zi):
    """
    this constructs a symmetry function as a sum of gaussians of
    pairwise displacements, namely
    Gi = sum_j_N ( exp( -width*( rij - rshift )**2 )
    """
    Gi = 0
    for j_atom in range(len(rij)):
        fc = cutoff_function(rij[i_atom][j_atom], Rc)
        Gi = Gi + fc * Zi[j_atom] * np.exp(-width * (rij[i_atom][j_atom]-rshift)**2)
        #print( j_atom , Gi )
    return Gi

def angular_gaussian(rij, theta, i_atom, width, shift, lam, Rc, atomic_num):
    Gi = 0
    for j_atom in range(len(rij)):
        fcij = cutoff_function(rij[i_atom][j_atom], Rc)
        if i_atom == j_atom:
            continue
        for k_atom in range(j_atom, len(rij)):
            if i_atom == k_atom or j_atom == k_atom:
                continue
            fcik = cutoff_function(rij[i_atom][k_atom], Rc)
            fcjk = cutoff_function(rij[j_atom][k_atom], Rc)
            gauss = np.exp(-width*(((rij[i_atom][j_atom]-shift)**2)
                                  +((rij[i_atom][k_atom]-shift)**2)
                                  +((rij[j_atom][k_atom]-shift)**2)))
            atom_function = atomic_num[j_atom] * atomic_num[k_atom]
            cutoff = fcij * fcik * fcjk
            Gi = Gi + cutoff * atom_function * gauss * (1 + lam * np.cos(theta[i_atom][j_atom][k_atom])) 
    return Gi

def experimental_symfun(rij,i_atom,Rc,a,b,c,d,power=1):
    """
    this is an experimental symmetry function predicted by a genetic algorithm.
    arguments for this symmetry function include only radial parameters and a cutoff.
    """
    gauss = 0
    dist = 0
    for j_atom in range( rij.shape[0]):
        fc = cutoff_function(rij[i_atom][j_atom],Rc)
        gauss = gauss + np.exp(np.sin(4.72384 + 1.1469267 * rij[i_atom][j_atom])**2)
        dist = dist + rij[i_atom][j_atom]
    Gi = np.sin(4.15066 + 0.45407*dist + gauss)**power
    return Gi
