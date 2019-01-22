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


def evaluate_model(model, inputdir, testing_files, results_name, testing_energy,
                   testing_elec, testing_exch, testing_ind, testing_disp):
    """Evaluate NN-SAPT model for accuracy and write the results."""
    print("Loading symmetry functions from file...\n")
    t1 = time.time()
    sym_dir = "../data/%s_sym_inp" % inputdir
    sym_input = []
    for i in range(
            len([
                name for name in os.listdir("%s" % sym_dir)
                if os.path.isfile(os.path.join(sym_dir, name))
            ])):
        sym_input.append(np.load(os.path.join(sym_dir, "symfun_%s.npy" % i)))

    sym_input = routines.scale_symmetry_input(sym_input)
    sym_input = np.array(sym_input)
    sym_input = np.transpose(sym_input, (1, 0, 2))
    sym_input = list(sym_input)
    t2 = time.time()
    elapsed = math.floor(t2 - t1)
    print("Symmetry functions loaded in %s seconds\n" % elapsed)

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
    with open('../results/%s.csv' % (results_name), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()
    return

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    inputdir="NMe-acetamide_Don--Aniline"
    
    print("Collecting properties and geometries...\n")
    t1 = time.time()
    NNff = NNforce_field('GA_opt',0,0)
    data_obj = routines.DataSet("../data/Final_Sampling_Alex/%s_Step_4_AlexSampling.csv"%inputdir,"xyzdir","alex","kcal/mol")
    (testing_aname,testing_atom_tensor,testing_xyz,
                    testing_elec,testing_ind,testing_disp,
                    testing_exch,testing_energy,
                    testing_geom_files,aname,atom_tensor,xyz,
                    elec,ind,disp,exch,
                    energy,geom_files) = data_obj.read_from_target_list()
    
    model = load_model("%s_model.h5"%inputdir)
    #for i in range(len(aname[0])):
    #    print(aname[0][i])
    
    t2 = time.time()
    elapsed = math.floor(t2-t1)
    print("Properties and geometries collected in %s seconds\n"%elapsed)
