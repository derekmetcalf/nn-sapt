import sys
import math
import numpy as np
from multiprocessing import pool
from numpy import linalg as LA
"""
these are symmetry functions used as
input to the neural network
"""


def cutoff_function(rmag, Rc):
    """Compute cutoff function for Gaussian symmetry functions."""
    if (rmag <= Rc):
        fc = 0.5 * (np.cos(np.pi * rmag / Rc) + 1)
    else:
        fc = 0.0
    return fc

def radial_gaussian(rij, i_atom, width, rshift, Rc, Zi):
    """Construct radial symmetry function as a sum of weighted Gaussians."""
    Gi = 0
    for j_atom in range(len(rij)):
        fc = cutoff_function(rij[i_atom][j_atom], Rc)
        Gi = Gi + fc * Zi[j_atom] * np.exp(-width *
                                           (rij[i_atom][j_atom] - rshift)**2)
    return Gi


def angular_gaussian(rij, theta, i_atom, width, shift, 
                        lam, Rc, atomic_num):
    """Construct angular symmetry function."""
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
            gauss = np.exp(-width * (((rij[i_atom][j_atom] - shift)**2) + (
                (rij[i_atom][k_atom] - shift)**2) + (
                    (rij[j_atom][k_atom] - shift)**2)))
            atom_function = atomic_num[j_atom] * atomic_num[k_atom]
            cutoff = fcij * fcik * fcjk
            Gi = Gi + cutoff * atom_function * gauss * (
                1 + lam * np.cos(theta[i_atom][j_atom][k_atom]))
    return Gi


def split_radial_gaussian(rij, i_atom, width, rshift, Rc, Zi, 
                        split_val=None, cross_terms=False):
    """Construct radial symmetry function as a sum of weighted Gaussians."""
    Gi = 0
    for j_atom in range(len(rij)):
        if i_atom<split_val:
            if cross_terms==False:
                if j_atom>=split_val: continue
                elif j_atom<split_val: pass
            elif cross_terms==True:
                if j_atom>=split_val: pass
                elif j_atom<split_val: continue
        elif i_atom>=split_val:
            if cross_terms==False:
                if j_atom>=split_val: pass
                elif j_atom<split_val: continue
            elif cross_terms==True:
                if j_atom>=split_val: continue
                elif j_atom<split_val: pass

        fc = cutoff_function(rij[i_atom][j_atom], Rc)
        Gi = Gi + fc * Zi[j_atom] * np.exp(-width *
                                           (rij[i_atom][j_atom] - rshift)**2)
    return Gi


def split_angular_gaussian(rij, theta, i_atom, width, shift, 
                        lam, Rc, atomic_num, split_val=None, cross_terms=False):
    """Construct angular symmetry function."""
    Gi = 0
    for j_atom in range(len(rij)):
        fcij = cutoff_function(rij[i_atom][j_atom], Rc)
        if i_atom == j_atom:
            continue
        for k_atom in range(j_atom, len(rij)):
            if i_atom == k_atom or j_atom == k_atom:
                continue
            if i_atom<split_val:
                if cross_terms==False:
                    if j_atom>=split_val or k_atom>=split_val: continue
                    else: pass
                elif cross_terms==True:
                    if j_atom<split_val and k_atom<split_val: continue
                    else: pass
            if i_atom>=split_val:
                if cross_terms==False:
                    if j_atom<split_val or k_atom<split_val: continue
                    else: pass
                elif cross_terms==True:
                    if j_atom>=split_val and k_atom>=split_val: continue
                    else: pass
            fcik = cutoff_function(rij[i_atom][k_atom], Rc)
            fcjk = cutoff_function(rij[j_atom][k_atom], Rc)
            gauss = np.exp(-width * (((rij[i_atom][j_atom] - shift)**2) + (
                (rij[i_atom][k_atom] - shift)**2) + (
                    (rij[j_atom][k_atom] - shift)**2)))
            atom_function = atomic_num[j_atom] * atomic_num[k_atom]
            cutoff = fcij * fcik * fcjk
            Gi = Gi + cutoff * atom_function * gauss * (
                1 + lam * np.cos(theta[i_atom][j_atom][k_atom]))
    return Gi

