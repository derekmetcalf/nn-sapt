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

    if uncertainty == False:
        sym_input = np.transpose(sym_input, (1, 0, 2))
        sym_input = list(sym_input)
        (energy_pred, elec_pred, exch_pred, ind_pred,
         disp_pred) = routines.infer_on_test(model, sym_input)
    else:
        sym_input = list(sym_input)
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
    print(energy_dist)
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
    #print(np.average(molec_stds))
    #graph_en = molec_ens[molec_id]
    
    connectivity = routines.get_connectivity_mat(xyz[molec_id])

    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111, projection='3d')
    #print(len(xyz))
    #print(np.array(xyz).shape)
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

def scale_sym_input(sym_input):
    bias = [0.045121926845448836, 0.06685110826116826, 0.09854527719319243, 0.14302176306429296, 0.21642438688952337, 0.32187628153048925, 0.4383189538183894, 0.5881807544207095, 0.8552761217872247, 1.3020722321665452, 2.0212418508489907, 2.715372942029979, 3.086571956122772, 3.6687917523235125, 4.417917017290391, 4.951150025832645, 5.44453952620702, 7.4563305639949204, 10.304709814523116, 10.052600677704403, 6.7211839345073905, 5.8321970133750085, 7.220880143596443, 6.026042131945646, 2.6460355404113884, 1.2007967458040005, 47.7884604069902, 159.0398946829241, 15.440710812324745, 51.54779809439906, 0.5931149418911059, 2.242997311162061, 0.9754634324116452, 1.290634433884927, 1.64979419538944, 2.052906213195718, 2.4951055209147923, 2.9469728476657187, 3.364018283680732, 3.7223344625358084, 3.9932648399458865, 4.134850714585474, 4.117112371805091, 3.943899900744897, 3.6523230385428653, 3.248526643187187, 2.7465546881221994, 2.215764923916782, 1.6793462258101315, 1.2294563551009985, 0.9424436175592992, 0.7265049730765343, 0.41586802177187154, 0.21927548737709152, 0.2671969144911964, 0.3040325871821045, 0.15571040808450912, 0.031125876009223837, 11.714311012926304, 30.468282157048613, 1.6187855463883165, 4.687828724891707, 0.03314812970112537, 0.1429160755001379, 0.0,0.0,0.0,0.0,0.0,0.0]
    multiplier = [0.28660264145046155, 0.3578880171705309, 0.46293581482017016, 0.5763662499179967, 0.7557747103030364, 0.9990966455568058, 1.186806417195226, 1.3361018401136686, 1.5602686699469384, 1.9904074398185212, 2.4100409600168446, 2.721615643470005, 2.8706622516870994, 3.3028836943328046, 3.6135238926378555, 3.6234688545375464, 3.5773763132846765, 4.544337357801568, 5.239031195056098, 4.08006130463718, 2.5578865552735954, 3.634104942090687, 4.598173103694901, 2.1001333309813996, 1.105193068691358, 0.4338102366344055, 41.86175085355946, 70.01266649525438, 13.950241836231735, 20.995563183019822, 0.5071176566641669, 0.9099538517258839, 1.2926737903738574, 1.531033799230931, 1.76542568677192, 1.9934027028468109, 2.2175599093777993, 2.4447519325987805, 2.6568158122425505, 2.8542641257798524, 3.0671630253810194, 3.310415496644588, 3.5653719673543187, 3.787512257144543, 3.9403482716435643, 3.9804336361808343, 3.8268752308726603, 3.529347294718805, 3.074632957193901, 2.624065865325729, 2.3536003388052125, 2.0874034001266066, 1.3444608478189422, 0.8469004915052069, 1.0738045733034947, 1.224858556736862, 0.6573640992747167, 0.13754697496505225, 18.408779237766748, 40.47036267926514, 3.8568557517840945, 10.938339696463233, 0.12189461404312532, 0.5152256228205875,1.0,1.0,1.0,1.0,1.0,1.0] 
    print(len(bias))
    print(np.array(sym_input).shape)
    for i in range(len(sym_input)):
        for j in range(len(sym_input[i])):
            for k in range(len(sym_input[i][j])):
                sym_input[i][j][k] = (sym_input[i][j][k]-bias[k])/multiplier[k]
    return sym_input

if __name__ == "__main__":
    """
    Choose target directories and test out models.
    Get errors and/or graph.

    """

    # Choose symmetry functions used for generation
    NNff = NNforce_field('GA_opt',0,0)
    
    # Choose dirs used for training of these models (this could be automated)
    #inputdirs = ["../data/random-aniline-nma-bare"]
    inputdirs = ["./SSI_neutral"]
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
    
    paths = []#insert relevant paths here     

    for num, path in enumerate(paths):
            # Gather test SAPT and descriptor info from path
        (filenames, en_ph, elst_ph, exch_ph, ind_ph, disp_ph, vs_ph) = routines.get_sapt_from_combo_files(path)
        (atoms, atom_nums, xyz) = routines.get_xyz_from_combo_files(path, filenames)
        (filenames,tot_en,elst,exch,ind,disp,split_vec) = routines.get_sapt_from_combo_files(path)
        (test_ntype, test_atype, test_unique_atoms) = routines.create_atype_list(atoms,routines.atomic_dictionary())

        sym_input = []
        for i in range(len(filenames)):
            file = f"{path}/{filenames[i]}_symfun.npy"
            sym_input.append(np.load(file, allow_pickle=True))
            print(sym_input[i].shape)
        mask = routines.get_test_mask(atom_nums, train_atom_nums)
        mask = routines.pad_sym_inp(mask, max_train_atoms=max_train_atoms)
        sym_input = routines.pad_sym_inp(sym_input, max_train_atoms=max_train_atoms)
        sym_input = np.concatenate((sym_input,mask),axis=2)
        sym_input = scale_sym_input(sym_input)

        modelname = "test_model"

        model = load_model(f"./{modelname}_model.h5")

        # Choose results path
        results_name = f"./test_model_results"

        xyz = routines.pad_sym_inp(xyz)

        # Evaluate model and save results to {results_name}.csv
        evaluate_model(model, sym_input, filenames, results_name, tot_en,
                            elst, exch, ind, disp, uncertainty=True)
