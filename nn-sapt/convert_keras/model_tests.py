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
import routines
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
import sapt_net
from symfun_parameters import *

"""Evaluate performance of NN-SAPT models.

This is a "runtime-maintained" file whose contents depends highly on the
nature of the required evaluation. Do not run this blindly.

"""


def evaluate_model(model, sym_input, testing_files, results_name, testing_energy,
                   testing_elec, testing_exch, testing_ind, testing_disp):
    """Evaluate NN-SAPT model for accuracy and write the results."""
    
    sym_input = np.array(sym_input)
    sym_input = np.transpose(sym_input, (1, 0, 2))
    sym_input = list(sym_input)

    (energy_pred, elec_pred, exch_pred, ind_pred,
     disp_pred) = routines.infer_on_test(model, sym_input)

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
    line = [
        'file', 'energy', ' energy_pred', ' elec', ' elec_pred', ' exch',
        ' exch_pred', ' ind', ' ind_pred', ' disp', ' disp_pred'
    ]
    csv_file.append(line)
    for i in range(len(energy_pred)):
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
        csv_file.append(line)
    with open('./%s.csv' % (results_name), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()
    return

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    NNff = NNforce_field('GA_opt',0,0)
    
    #path = "../data/NMe-acetamide_Don--Aniline_crystallographic"
    path = "./shuffled_xyzs"
     
    (filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)
   
    sym_input = [] 
    for i in range(len(filenames)):
        file = f"{path}/{filenames[i]}_symfun.npy"
        sym_input.append(np.load(file)) 
    
    filenames.append("zero_boye")
    tot_en.append(0)
    elst.append(0)
    exch.append(0)
    ind.append(0)
    disp.append(0)
    sym_input.append(np.zeros(np.shape(sym_input[0]))) 
    

    model = load_model("./NMA-Aniline-0.1-pert_model.h5")
     
    results_name = "NMA_Aniline_test"
    
    evaluate_model(model, sym_input, filenames, results_name, tot_en,
                        elst, exch, ind, disp) 
