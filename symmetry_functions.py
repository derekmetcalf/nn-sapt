import sys
import numpy as np
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


def radial_gaussian( rij, i_atom , width, rshift, Rc ):
    """
    this constructs a symmetry function as a sum of gaussians of
    pairwise displacements, namely
    Gi = sum_j_N ( exp( -width*( rij - rshift )**2 )
    """

    #print(" symmetry function ", i_atom )

    Gi=0
    for j_atom in range( rij.shape[0] ):

        fc = cutoff_function( rij[i_atom][j_atom] , Rc )
        Gi = Gi + fc * np.exp(-width * (rij[i_atom][j_atom]-rshift)**2 )
        #print( j_atom , Gi )

    return Gi
