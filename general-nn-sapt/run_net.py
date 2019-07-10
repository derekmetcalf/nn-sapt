from __future__ import print_function
import os
import csv
import sys
import math
import time
import glob
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, Dropout, Concatenate, concatenate, add
from keras.utils import plot_model
from keras import regularizers
from sys import stdout
from sklearn import preprocessing
import routines
import model_tests
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
import tensorflow as tf
import sapt_net
import model_tests
from symfun_parameters import *

"""Train one or many neural networks.

This document is "runtime-maintained" and care should be taken in blindly 
running it. It may be less time-consuming for a NN-SAPT novice to write 
their own run script for their purposes.
 
"""


def standard_run(sym_input,aname,inputdir,geom_files,energy,elec,exch,
                ind,disp,n_layer,nodes,mask,mt_vec,val_split,
                dropout_fraction,l2_reg,epochs, results_name):
    """
    Train Behler-Parrinello neural network with provided symmetry function
    input, labels, network architecture and architecture. Output trained
    models and write training results to .csv .

    """
    (model, test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
     test_disp, disp_pred, test_ind, ind_pred, avg_mae,
     std_mae, atom_model) = sapt_net.sapt_net(
         sym_input,
         aname,
         inputdir,
         energy,
         elec,
         exch,
         ind,
         disp,
         n_layer,
         nodes,
         multitarget_vec=mt_vec,
         val_split=val_split,
         data_fraction=(1 - val_split),
         dropout_fraction=dropout_fraction,
         l2_reg=l2_reg,
         epochs=epochs)
    (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err, exch_mae,
     exch_rmse, exch_max_err, disp_mae, disp_rmse, disp_max_err,
     ind_mae, ind_rmse, ind_max_err) = sapt_net.sapt_errors(
         test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
         test_disp, disp_pred, test_ind, ind_pred)
    
    csv_file = []
    line = ["mae", "rmse", "max_error"]
    csv_file.append(line)
    line = ["total", en_mae, en_rmse, en_max_err]
    csv_file.append(line)
    line = ["elst", elec_mae, elec_rmse, elec_max_err]
    csv_file.append(line)
    line = ["exch", exch_mae, exch_rmse, exch_max_err]
    csv_file.append(line)
    line = ["ind", ind_mae, ind_rmse, ind_max_err]
    csv_file.append(line)
    line = ["disp", disp_mae, disp_rmse, disp_max_err]
    csv_file.append(line)
    line = [
        'files', 'energy', ' energy_pred', ' elec', ' elec_pred', ' exch',
        ' exch_pred', ' ind', ' ind_pred', ' disp', ' disp_pred'
    ]
    csv_file.append(line)
    for i in range(len(energy_pred)):
        line = [
            geom_files[i], test_en[i], energy_pred[i], test_elec[i],
            elec_pred[i], test_exch[i], exch_pred[i], test_ind[i], ind_pred[i],
            test_disp[i], disp_pred[i]
        ]
        csv_file.append(line)
    with open('%s_results.csv' % (inputdir), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()
    
    return model, atom_model

if __name__ == "__main__":
    """
    Choose model training characteristics and execute. 
    TODO: Wrap preprocessing as a class? Very ugly currently

    """ 
    
    # Choose paths from which to extract input directories or list
    # directories explicitly. These will contain dm-xyzs and .npy symfuns 
    #paths = ["./script_tests"]
    inputdirs = ["./script_tests"]
    #inputdirs = []
    #for path in paths:
    #    for r,d,f in os.walk(path):
    #        for folder in d:
    #            inputdirs.append(os.path.join(r,folder))
    
    # Extract necessary info for training from dm-xyz and symfun files
    aname = []
    atom_tensor = []
    xyz = []
    elst = []
    ind = []
    disp = []
    exch = []
    tot_en = []
    geom_files = []
    sym_input = []
    symfun_files = []
    atom_nums = []
    atoms = []
    filenames = []
    for i in range(len(inputdirs)):

        # The following are custom choices for training on SSI directories
        # and fractions of non-SSI directories via the keep_prob argument

        path = inputdirs[i]
        print("Collecting properties and geometries...\n")
        t1 = time.time()
        #if "SSI_spiked" in path:
        #    (set_filenames,set_tot_en,set_elst,set_exch,set_ind,set_disp,val_split) = routines.get_sapt_from_combo_files(path, keep_prob=1.00)
        #    (set_atoms,set_atom_nums,set_xyz) = routines.get_xyz_from_combo_files(path,set_filenames)
        #else:
        #    (set_filenames,set_tot_en,set_elst,set_exch,set_ind,set_disp,val_split) = routines.get_sapt_from_combo_files(path, keep_prob=0.0125)
        #    (set_atoms,set_atom_nums,set_xyz) = routines.get_xyz_from_combo_files(path,set_filenames)
        
        
        (set_filenames,set_tot_en,set_elst,set_exch,set_ind,set_disp,val_split) = routines.get_sapt_from_combo_files(path, keep_prob=1.00)
        (set_atoms,set_atom_nums,set_xyz) = routines.get_xyz_from_combo_files(path,set_filenames)

        for k in range(len(set_atom_nums)):
            atom_nums.append(set_atom_nums[k])
            atoms.append(set_atoms[k])
            filenames.append(set_filenames[k])
            tot_en.append(set_tot_en[k])
            elst.append(set_elst[k])
            exch.append(set_exch[k])
            ind.append(set_ind[k])
            disp.append(set_disp[k])
            xyz.append(set_xyz[k])

        t2 = time.time()
        elapsed = math.floor(t2 - t1)
        print("Properties and geometries collected in %s seconds\n" % elapsed)
        
        print("Loading symmetry functions from file...\n")
        t1 = time.time()


        for j in range(len(set_filenames)):
            file = f"{path}/{set_filenames[j]}_symfun.npy"
            symfun_files.append(file)
            sym_input.append(np.load(file, allow_pickle=True))


        t2 = time.time()
        elapsed = math.floor(t2 - t1)
        print("Symmetry functions loaded in %s seconds\n" % elapsed)
    print(f"atom_nums len: {len(atom_nums)}")
    print(f"atom_nums[0]: {atom_nums[0]}")
    mask = routines.get_mask(atom_nums)
    mask = routines.pad_sym_inp(mask)
    print(f"mask shape: {mask.shape}")
    sym_input = np.array(sym_input) 
    sym_input = routines.pad_sym_inp(sym_input)
    
    sym_input = np.concatenate((sym_input, mask), axis=2)
    print(f"sym inp shape: {sym_input.shape}")
    (ntype, atype, unique_atoms) = routines.create_atype_list(atoms,routines.atomic_dictionary()) 

    # Define training parameters
    dropout_fraction = 0.05
    l2_reg = 0.005
    nodes = [100,100,75]
    val_split = 0.10
    epochs = 10
    n_layer = len(nodes)
    mt_vec = [0.6, 0.1, 0.1, 0.1, 0.1]
    
    # Choose output name
    results_name = "script_test_model_out"

    # Train models
    (model, atom_model) = standard_run(sym_input,atoms,results_name,
                    filenames,tot_en,elst,exch,
                    ind,disp,n_layer,nodes,mask,mt_vec,val_split,
                    dropout_fraction,l2_reg,epochs,results_name)
    

    ## The following is written for training many MTP models
    #for j in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #    mt_vec = [1 - j, j / 4, j / 4, j / 4, j / 4]
        #results_name = "%s_tot_en_frac_%.1g"%(inputdir,float(mt_vec[0]))
    #    results_name = "NMA_Indole_Aniline_MeOH_combo%.1g"%mt_vec[0]
    #    standard_run(sym_input,aname,results_name,geom_files,energy,
    #                 elec,exch,ind,disp,n_layer,nodes,mt_vec,val_split,   
    #                 dropout_fraction,l2_reg,epochs,results_name)
