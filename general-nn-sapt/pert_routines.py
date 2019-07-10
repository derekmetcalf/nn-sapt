import os
import sys
import glob
import numpy as np
import time
from numpy import linalg as LA
from joblib import Parallel, delayed
import tensorflow as tf

def get_xyz_from_combo_files(path):
    
    atom_dict = atomic_dictionary()
    xyz = []
    atoms = []
    atom_nums = []
    for file in glob.glob("%s/*.xyz"%path):
        with open(file, "r") as xyzfile:
            next(xyzfile)
            next(xyzfile)
            molec_atoms = []
            molec_xyz = []
            molec_atom_nums = []
            for line in xyzfile:
                data = line.split()
                molec_atoms.append(data[0])
                molec_atom_nums.append(atom_dict[data[0]])
                molec_xyz.append([float(data[1]), float(data[2]), float(data[3])])
            atoms.append(molec_atoms)
            xyz.append(molec_xyz)
            atom_nums.append(molec_atom_nums)
    return atoms, atom_nums, xyz

def get_sapt_from_combo_files(path):
    """Collect SAPT energies and corresponding filenames from the combo xyz
    file / energy files written by routines.combine_energies_xyz

    """
    filenames = []
    tot_en = []
    elst = []
    exch = []
    ind = []
    disp = []
    for file in glob.glob("%s/*.xyz"%path):
        with open(file, "r") as saptfile:
            temp = saptfile.readlines()
            data = temp[1].split(",")
            filenames.append(data[0])
            tot_en.append(float(data[1]))
            elst.append(float(data[2]))
            exch.append(float(data[3]))
            ind.append(float(data[4]))
            disp.append(float(data[5])) 
    return filenames, tot_en, elst, exch, ind, disp    

def make_perturbed_xyzs(path, max_shift=0.02):
    """Currently, make a perturbed xyz file for each file in the target
    directory and deposits those files in the same path.

    """
    (atoms,atom_nums,xyz) = get_xyz_from_combo_files(path)
    (filenames,tot_int,elst,exch,ind,disp) = get_sapt_from_combo_files(path)
    for i in range(len(xyz)):
        perturbation = (np.random.rand(len(xyz[i]),3) - 0.5) * 2 * max_shift
        new_xyz = xyz[i] + perturbation
        new_filename = "supp_" + filenames[i]
        with open("%s/%s"%(path,new_filename),"w+") as xyz_file:
            xyz_file.write("%s\n"%len(new_xyz))
            xyz_file.write("%s,%s,%s,%s,%s,%s\n"%(new_filename,tot_int[i],elst[i],exch[i],ind[i],disp[i]))
            for j in range(len(new_xyz)):
                xyz_file.write("%s %s %s %s\n"%(atoms[i][j],new_xyz[j,0],new_xyz[j,1],new_xyz[j,2]))
    return

def atomic_dictionary():
    """Construct dictionary associating atoms with their atomic numbers.

    Lol there has to be a package that does this but I guess I'm good until
    we need Cesium.

    """
    atom_dict = {
        'H': 1,
        'He': 2,
        'Li': 3,
        'Be': 4,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Ne': 10,
        'Na': 11,
        'Mg': 12,
        'Al': 13,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17,
        'Ar': 18,
        'K': 19,
        'Ca': 20,
        'Sc': 21,
        'Ti': 22,
        'V': 23,
        'Cr': 24,
        'Mn': 25,
        'Fe': 26,
        'Co': 27,
        'Ni': 28,
        'Cu': 29,
        'Zn': 30,
        'Ga': 31,
        'Ge': 32,
        'As': 33,
        'Se': 34,
        'Br': 35,
        'Kr': 36,
        'Rb': 37,
        'Sr': 38,
        'Y': 39,
        'Zr': 40,
        'Nb': 41,
        'Mo': 42,
        'Tc': 43,
        'Ru': 44,
        'Rh': 45,
        'Pd': 46,
        'Ag': 47,
        'Cd': 48,
        'In': 49,
        'Sn': 50,
        'Sb': 51,
        'Te': 52,
        'I': 53,
        'Xe': 54
    }
    return atom_dict
