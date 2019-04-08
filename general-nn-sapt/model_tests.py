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
nature of the required evaluation. Do not run this blindly.

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
    
    graph_en = molec_ens[molec_id]
    
    connectivity = routines.get_connectivity_mat(xyz[molec_id])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xyz = np.transpose(np.array(xyz),(2,0,1))
    graph_std = molec_stds[molec_id][:len(xyz[0][molec_id])]
    graph_en = molec_ens[molec_id][:len(xyz[0][molec_id])]

    if prop == "energy": prop=graph_en
    elif prop == "uncertainty": prop=graph_std
    else: print("invalid property for graphing"); quit()
    
    sc = ax.scatter(xyz[0][molec_id], xyz[1][molec_id], xyz[2][molec_id], s=150, c=graph_std, cmap="YlOrBr", alpha=1, zorder=2)#, label=lab)
    for i in range(len(atoms[molec_id])):
        txt = ax.text(xyz[0][molec_id][i]+0.1, xyz[1][molec_id][i]+0.1, xyz[2][molec_id][i]+0.1, atoms[molec_id][i], size=20, zorder=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='w')])
    plt.draw()    
    
    for i in range(len(connectivity)):
        for j in range(len(connectivity[i])):
            if connectivity[i][j] == 1:
                ax.plot([xyz[0][molec_id][i],xyz[0][molec_id][j]],[xyz[1][molec_id][i],xyz[1][molec_id][j]],[xyz[2][molec_id][i],xyz[2][molec_id][j]], c="xkcd:black", zorder=1)

    plt.colorbar(sc)
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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    NNff = NNforce_field('GA_opt',0,0)
 
    train_path = "./SSI_neutral"
    (train_atoms, train_atom_nums, train_xyz) = routines.get_xyz_from_combo_files(train_path)
    max_train_atoms = 0
    for i in range(len(train_atoms)):
        if len(train_atoms[i]) > max_train_atoms:
            max_train_atoms = len(train_atoms[i])
    
    path = "./NMA-Aniline-crystallographic_sym_inp"
    
    (atoms, atom_nums, xyz) = routines.get_xyz_from_combo_files(path)
    (filenames,tot_en,elst,exch,ind,disp,split_vec) = routines.get_sapt_from_combo_files(path)
    
    sym_input = []
    for i in range(len(filenames)):
        file = f"{path}/{filenames[i]}_symfun.npy"
        sym_input.append(np.load(file))
    mask = routines.get_test_mask(atom_nums, train_atom_nums)
    mask = routines.pad_sym_inp(mask, max_train_atoms=max_train_atoms)
    sym_input = routines.pad_sym_inp(sym_input, max_train_atoms=max_train_atoms)
    sym_input = np.concatenate((sym_input,mask),axis=2)

    model = load_model("./SSI_neutral_model.h5")
    atom_model = load_model("./SSI_neutral_atomic_model.h5")
    results_name = "dropout_uncertainty_SSI_only"
       
    molec_id = 12 #which sample from dataset to graph
    molecular_viewer(atom_model, sym_input, xyz, molec_id, prop="uncertainty")   
    
    #evaluate_model(model, sym_input, filenames, results_name, tot_en,
    #                    elst, exch, ind, disp, uncertainty=True)