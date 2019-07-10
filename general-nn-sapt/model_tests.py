from __future__ import print_function
import os
import csv
import sys
import math
import time
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Lambda, Dense, Activation, Dropout, concatenate, add
from keras.utils import plot_model
from keras import regularizers
from sys import stdout
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import routines
import matplotlib
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
import sapt_net
from symfun_parameters import *

"""Evaluate performance of NN-SAPT models.

This is a "runtime-maintained" file whose contents depends highly on the
nature of the required evaluation. Do not run blindly.

"""


def evaluate_model(model, sym_input, testing_files, results_name, 
                   testing_energy, testing_elec, testing_exch, 
                   testing_ind, testing_disp, uncertainty=False):
    """Evaluate NN-SAPT model for accuracy and write the results."""
    


    sym_input = np.array(sym_input)
    sym_input = np.transpose(sym_input, (1, 0, 2))
    sym_input = list(sym_input)
    
    if uncertainty == False:
        (energy_pred, elec_pred, exch_pred, ind_pred,
         disp_pred) = routines.infer_on_test(model, sym_input)
    else:
        (energy_pred, elec_pred, exch_pred,
         ind_pred, disp_pred, energy_sig,
         elec_sig, exch_sig, ind_sig,
         disp_sig) = uncertainty_inferences(model, sym_input)
    print("Inferred on inputs")

    energy_pred = np.array(np.array(energy_pred).T[0])
    elec_pred = np.array(np.array(elec_pred).T[0])
    exch_pred = np.array(np.array(exch_pred).T[0])
    ind_pred = np.array(np.array(ind_pred).T[0])
    disp_pred = np.array(np.array(disp_pred).T[0])
    print(energy_pred)

    (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err, exch_mae,
     exch_rmse, exch_max_err, disp_mae, disp_rmse, disp_max_err,
     ind_mae, ind_rmse, ind_max_err) = sapt_net.sapt_errors(
         testing_energy, energy_pred, testing_elec, elec_pred, testing_exch,
         exch_pred, testing_disp, disp_pred, testing_ind, ind_pred)
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
    
    if uncertainty == False:
        line = [
        'file', 'energy', ' energy_pred', ' elec', ' elec_pred', ' exch',
        ' exch_pred', ' ind', ' ind_pred', ' disp', ' disp_pred'
        ]
    else:
        line = [
        'file', ' energy', ' energy_pred', ' energy_sig',' elec', 
        ' elec_pred', ' elec_sig', ' exch', ' exch_pred', ' exch_sig',
        ' ind', ' ind_pred', ' ind_sig',
        ' disp', ' disp_pred', ' disp_sig'
        ]
    csv_file.append(line)
    for i in range(len(energy_pred)):
        if uncertainty == False:
            line = [
                testing_files[i],
                float(testing_energy[i]),
                float(energy_pred[i]),
                float(testing_elec[i]),
                float(elec_pred[i]),
                float(testing_exch[i]),
                float(exch_pred[i]),
                float(testing_ind[i]),
                float(ind_pred[i]),
                float(testing_disp[i]),
                float(disp_pred[i])
            ]
        else:
            line = [
                testing_files[i],
                float(testing_energy[i]),
                float(energy_pred[i]),
                float(energy_sig[i]),
                float(testing_elec[i]),
                float(elec_pred[i]),
                float(elec_sig[i]),
                float(testing_exch[i]),
                float(exch_pred[i]),
                float(exch_sig[i]),
                float(testing_ind[i]),
                float(ind_pred[i]),
                float(ind_sig[i]),
                float(testing_disp[i]),
                float(disp_pred[i]),
                float(disp_sig[i])
            ]
        csv_file.append(line)
    with open('./%s.csv' % (results_name), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()
    return

def uncertainty_inferences(model, sym_input, simulations=1000):
    """Uncertainty estimates via Monte Carlo dropout uncertainty"""
    sym_input = np.array(sym_input)
    sym_input = np.transpose(sym_input, (1, 0, 2))
    sym_input = list(sym_input)
    energy_dist = []
    elec_dist = []
    exch_dist = []
    ind_dist = []
    disp_dist = []
    for i in range(simulations):
        (energy_single, elec_single, exch_single, ind_single,
         disp_single) = routines.infer_on_test(model, sym_input)
        energy_dist.append(energy_single)
        elec_dist.append(elec_single)
        exch_dist.append(exch_single)
        ind_dist.append(ind_single)
        disp_dist.append(disp_single)
    energy_pred = np.average(energy_dist,axis=0)
    elec_pred = np.average(elec_dist,axis=0)
    exch_pred = np.average(exch_dist,axis=0)
    ind_pred = np.average(ind_dist,axis=0)
    disp_pred = np.average(disp_dist,axis=0)
    energy_sig = np.std(energy_dist,axis=0)
    elec_sig = np.std(elec_dist,axis=0)
    exch_sig = np.std(exch_dist,axis=0)
    ind_sig = np.std(ind_dist,axis=0)
    disp_sig = np.std(disp_dist,axis=0)
    return (energy_pred, elec_pred, exch_pred,
         ind_pred, disp_pred, energy_sig,
         elec_sig, exch_sig, ind_sig,
         disp_sig)

def molecular_viewer(atom_model, sym_input, xyz, molec_id, prop="energy"):    
    atom_output, atom_std = get_atom_outs(atom_model, sym_input)
    atom_output = np.transpose(atom_output, (1,0,2,3))
    atom_std = np.transpose(atom_std, (1,0,2,3))
    molec_ens = []
    molec_stds = []
    for molec in range(len(atom_output)):
        atom_ens = []
        atom_stds = []
        for atom in range(len(atom_output[molec])):
            atom_contrib = np.sum(atom_output[molec][atom])
            atom_ens.append(atom_contrib)
            atom_std_contrib = np.sum(atom_std[molec][atom])
            atom_stds.append(atom_std_contrib)
        molec_ens.append(atom_ens)
        molec_stds.append(atom_stds)
    molec_ens = np.array(molec_ens)
    molec_stds = np.array(molec_stds)
    print(np.average(molec_stds))
    #graph_en = molec_ens[molec_id]
    
    connectivity = routines.get_connectivity_mat(xyz[molec_id])

    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111, projection='3d')
    #xyz = np.transpose(np.array(xyz),(1,0))
    xyz = np.transpose(np.array(xyz),(2,0,1))
    """
    xs = []
    ys = []
    zs = []
    for i in range(len(xyz)):
        for j in range(len(xyz[i])):
            for k in range(len(xyz[i][j])):
                xs.append(xyz
                ys.append(xyz
                zs.append(xyz
    """
    #print(xyz[0][molec_id])
    graph_std = molec_stds[molec_id][:len(xyz[0][molec_id])]
    graph_en = molec_ens[molec_id][:len(xyz[0][molec_id])]
    print(f"Interaction Energy: {np.sum(graph_en)}")
    print(f"Uncertainty: {np.sum(graph_std)}")
    if prop == "energy":
        prop=graph_en; vmin=-0.75; vmax=0.75; cmap=matplotlib.cm.RdBu_r
    elif prop == "uncertainty": 
        prop=graph_std; vmin=0; vmax=2; cmap="YlOrBr"
    else: print("invalid property for graphing"); quit()
    print(xyz[0][molec_id]) 
    sc = ax.scatter(xyz[0][molec_id], xyz[1][molec_id], xyz[2][molec_id], s=150, c=prop, cmap=cmap, alpha=1, zorder=2, vmin=vmin, vmax=vmax)#, label=lab)
    for i in range(len(atoms[molec_id])):
        txt = ax.text(xyz[0][molec_id][i]+0.1, xyz[1][molec_id][i]+0.1, xyz[2][molec_id][i]+0.1, atoms[molec_id][i], size=20, zorder=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='w')])
    plt.draw()    
    
    for i in range(len(connectivity)):
        for j in range(len(connectivity[i])):
            if connectivity[i][j] == 1:
                ax.plot([xyz[0][molec_id][i],xyz[0][molec_id][j]],[xyz[1][molec_id][i],xyz[1][molec_id][j]],[xyz[2][molec_id][i],xyz[2][molec_id][j]], c="xkcd:black", zorder=1)
    plt.colorbar(sc,fraction=0.035)
    plt.axis('equal')
    plt.show()
    return

def get_atom_outs(atom_model, sym_input, simulations=1000):
    layer_name = "atom_comps/concat"
    sym_input = np.array(sym_input)
    sym_input = np.transpose(sym_input, (1, 0, 2))
    sym_input = list(sym_input)
    atom_sim = []
    for i in range(simulations):
        atom_sim.append(atom_model.predict_on_batch(sym_input))
    atom_output = np.average(atom_sim, axis=0)
    atom_std = np.std(atom_sim, axis=0)
    return atom_output, atom_std

if __name__ == "__main__":
    """
    Choose target directories and test out models.
    Get errors and/or graph.

    """
    
    # Choose symmetry functions used for generation
    NNff = NNforce_field('GA_opt',0,0)
    
    # Choose dirs used for training of these models (this could be automated)
    inputdirs = ["./new_SSI_spiked"]
    for r,d,f in os.walk("../data/5_28_19_pert-xyz-nrgs-derek-format"):
        for folder in d:
            inputdirs.append(os.path.join(r,folder))
    for r,d,f in os.walk("../data/pert-xyz-nrgs-acceptors-derek-format"):
        for folder in d:
            inputdirs.append(os.path.join(r,folder))
    #train_path = "./SSI_spiked_3"
    #train_path = "./SSI_spiked_2"
    
    # Parse training data to get pertinent information for testing
    train_atoms = []
    train_atom_nums = []
    train_xyz = []
    for i in range(len(inputdirs)):
        train_path = inputdirs[i]
        (filenames, en_ph, elst_ph, exch_ph, ind_ph, disp_ph, vs_ph) = routines.get_sapt_from_combo_files(train_path)
        (temp_atoms, temp_atom_nums, temp_xyz) = routines.get_xyz_from_combo_files(train_path, filenames)
        for j in range(len(temp_atoms)):
            train_atoms.append(temp_atoms[j])
            train_atom_nums.append(temp_atom_nums[j])
            train_xyz.append(temp_xyz[j])
    (train_ntype, train_atype, train_unique_atoms) = routines.create_atype_list(train_atoms,routines.atomic_dictionary())
    #train_unique_atoms.append('F')
    #(train_atoms, train_atom_nums, train_xyz) = routines.get_xyz_from_combo_files(train_path)
    max_train_atoms = 0
    for i in range(len(train_atoms)):
        if len(train_atoms[i]) > max_train_atoms:
            max_train_atoms = len(train_atoms[i])
    
    # Choose test path(s)
    for r,d,f in os.walk("../data/pert-xyz-nrgs-acceptors-derek-format-test"):
        for folder in d:
            path = os.path.join(r,folder)
            #print(path) 
            #path = "../data/20190413-test-molecules-10k-dm-format/NMA_Quinilone_random"
            #path = "../data/5_28_19_pert-xyz-nrgs-derek-format-test/Acc--NMe-acetamide_Don--1H-naphthalene-PLDB-89-dmformat-xyz"
        #path = "./SSI_neutral"
        
        # Gather test SAPT and descriptor info from path
        (filenames, en_ph, elst_ph, exch_ph, ind_ph, disp_ph, vs_ph) = routines.get_sapt_from_combo_files(path)
        (atoms, atom_nums, xyz) = routines.get_xyz_from_combo_files(path, filenames)
        (filenames,tot_en,elst,exch,ind,disp,split_vec) = routines.get_sapt_from_combo_files(path)
        (test_ntype, test_atype, test_unique_atoms) = routines.create_atype_list(atoms,routines.atomic_dictionary())
     
        sym_input = []
        for i in range(len(filenames)):
            file = f"{path}/{filenames[i]}_symfun.npy"
            sym_input.append(np.load(file, allow_pickle=True))
        mask = routines.get_test_mask(atom_nums, train_atom_nums)
        mask = routines.pad_sym_inp(mask, max_train_atoms=max_train_atoms)
        sym_input = routines.pad_sym_inp(sym_input, max_train_atoms=max_train_atoms)
        sym_input = np.concatenate((sym_input,mask),axis=2)
        
        # Load in desired model to test from .h5 files
        modelname = "neutral-SSI_0.0125-spike_100-100-75_retry"
        testset = path.split("/")[-1]

        model = load_model(f"./{modelname}_model.h5")
        atom_model = load_model(f"./{modelname}_atomic_model.h5")

        # Choose results name
        #results_name = f"./test_results/{modelname}{testset}"
        results_name = "SSI_preds"
        
        ## Get a molecular view of molec_id in test set, either predicted
        ## atomwise energy or atomwise uncertainty estimate
        #molec_id = 5 #which sample from dataset to graph

        #molecular_viewer(atom_model, sym_input, xyz, molec_id, prop="energy") 
        #molecular_viewer(atom_model, sym_input, xyz, molec_id, prop="uncertainty") 
        # Evaluate model and save results to {results_name}.csv

        sym_input = np.transpose(sym_input, (1,0,2)) 
        evaluate_model(model, sym_input, filenames, results_name, tot_en,
                            elst, exch, ind, disp, uncertainty=False)
