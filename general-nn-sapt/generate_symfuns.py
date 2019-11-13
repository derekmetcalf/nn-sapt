import os
import routines
import symfun_parameters
import numpy as np
import time
"""
Generates "intermolecular symmetry functions," a set of wACSFs for the same
monomer and another for the other monomer. Each generated exactly by the
Gastegger, et al prescription, with "cross-terms" arising from three-body
terms attributed to the intermolecular set.

"""

# Choose path from which to generate symfuns. Should include xyzs,
# with "filename,tot_interaction_en,elst,exch,ind,disp,atoms_in_mon_1" 
# as the comment line

path = "timing_test"

# Choose symmetry function parameters to use ("GA_opt" is from Gastegger)
# NN_A can contain different parameters than NN_B if different parameters
# are desired for the intra vs intermolecular part of the symfuns

NN_A = symfun_parameters.NNforce_field("GA_opt",0,0)
NN_B = symfun_parameters.NNforce_field("GA_opt",0,0)

# Dive paths within target. Can be adjusted depending on file structure.

folders = []
#for r,d,f in os.walk(path):
#    for folder in d:
#        folders.append(os.path.join(r,folder))

folders.append(path)

# Compute symfuns for all dm-xyzs in path

systems = 0
for f in folders:
    systems += 1

    # Get SAPT decomposition and intermolecular split locations
    (filenames,tot_en,elst,exch,ind,disp,split_vec) = routines.get_sapt_from_combo_files(f)

    # Get xyz coordinates

    (atoms,atom_nums,xyz) = routines.get_xyz_from_combo_files(f, filenames)
    
    # Compute requisite distance and angle 2- and 3- tensors
    time1 = time.time()

    routines.compute_displacements(xyz, f, filenames)
    routines.compute_thetas(len(xyz), f, filenames)

    # Compute symmetry function vector for each atom in each molecule
    
    routines.construct_symmetry_input(NN_A,f,filenames,
            atoms,np.zeros(len(xyz)), atom_nums,0,NN_B,split_vec)
    
    time2 = time.time()
    print(time2-time1)

