from __future__ import print_function
import os
import csv
import sys
import math
import time
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, Dropout, concatenate, add
from keras.utils import plot_model
from keras import regularizers
from sys import stdout
from sklearn import preprocessing
import routines
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
import sapt_net
import model_tests
from symfun_parameters import *

"""Train one or many neural networks.

This document is "runtime-maintained" and care should be taken in blindly 
running it. As a research tool, it is an exercise in bad Python and it may
be less time-consuming for a NN-SAPT novice to write their own run script
for their purposes.
 
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
inputdirs = ["NMe-acetamide_NMe-acetamide", "NMe-acetamide_Indole"]

sym_input = []
for i in range(len(inputdirs)):
    inputdir = inputdirs[i]
    print("Collecting properties and geometries...\n")
    t1 = time.time()
    NNff = NNforce_field('GA_opt', 0, 0)
    #(aname, xyz, elec, exch, ind, disp, energy, atom_tensor, en_units) = routines.read_sapt_data(inputdir)
    data_obj = routines.DataSet(
        "./Final_Sampling_Alex/%s_Step_4_AlexSampling.csv" % inputdir, "alex",
        "kcal/mol")
    (testing_aname, testing_atom_tensor, testing_xyz, testing_elec,
     testing_ind, testing_disp, testing_exch, testing_energy,
     testing_geom_files, aname, atom_tensor, xyz, elec, ind, disp, exch,
     energy, geom_files) = data_obj.read_from_target_list()
    #for i in range(len(aname[0])):
    #    print(aname[0][i])

    t2 = time.time()
    elapsed = math.floor(t2 - t1)
    print("Properties and geometries collected in %s seconds\n" % elapsed)

    print("Loading symmetry functions from file...\n")
    t1 = time.time()
    sym_dir = "%s_train_sym_inp" % inputdir
    for i in range(
            len([
                name for name in os.listdir("./%s" % sym_dir)
                if os.path.isfile(os.path.join(sym_dir, name))
            ])):
        sym_input.append(np.load(os.path.join(sym_dir, "symfun_%s.npy" % i)))

    sym_input = routines.scale_symmetry_input(sym_input)

    t2 = time.time()
    elapsed = math.floor(t2 - t1)
    print("Symmetry functions loaded in %s seconds\n" % elapsed)

sym_input = np.array(sym_input)

dropout_fraction = 0.05
l2_reg = 0.01
nodes = [300, 300]
val_split = 0.01
epochs = 100
n_layer = len(nodes)
mt_vec = [0.6, 0.1, 0.1, 0.1, 0.1]

#train 1 net as specified
if True:
    (model, test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
     test_disp, disp_pred, test_ind, ind_pred, avg_mae,
     std_mae) = sapt_net.sapt_net(
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
    with open('%s_comp_frac_0.csv' % (inputdir), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()

    #data_obj = routines.DataSet("./crystallographic_data/2019-01-07-CSD-NMA-NMA-xyz-nrgs-outs/FSAPT0/SAPT0-NRGS-COMPONENTS.txt",None,"kcal/mol")
    #(_aname,_atom_tensor,_xyz,testing_elec,testing_ind,testing_disp,testing_exch,
    #            testing_energy,geom_files) = data_obj.read_from_target_list()
    #print(geom_files)
    #model_tests.evaluate_model(model, "NMA-NMA-crystallographic", geom_files, testing_energy, testing_elec, testing_exch, testing_ind, testing_disp)

if False:
    for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        mt_vec = [1 - j, j / 4, j / 4, j / 4, j / 4]
        (test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
         test_disp, disp_pred, test_ind, ind_pred, avg_mae,
         std_mae) = sapt_net.sapt_net(
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
        (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err,
         exch_mae, exch_rmse, exch_max_err, disp_mae, disp_rmse, disp_max_err,
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
            'energy', ' energy_pred', ' elec', ' elec_pred', ' exch',
            ' exch_pred', ' ind', ' ind_pred', ' disp', ' disp_pred'
        ]
        csv_file.append(line)
        for i in range(len(energy_pred)):
            line = [
                test_en[i],
                np.float(energy_pred[i]), test_elec[i],
                np.float(elec_pred[i]), test_exch[i],
                np.float(exch_pred[i]), test_ind[i],
                np.float(ind_pred[i]), test_disp[i],
                np.float(disp_pred[i])
            ]
            csv_file.append(line)
        with open('%s_comp_frac_%s.csv' % (inputdir, int(j * 100)),
                  'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(csv_file)
        writeFile.close()

#saturation curve
if False:
    data_fraction = [0.8, 0.6, 0.4, 0.2, 0.1]

    for frac in data_fraction:
        (test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
         test_disp, disp_pred, test_ind, ind_pred, avg_mae,
         std_mae) = sapt_net.sapt_net(
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
             val_split=(1 - frac),
             data_fraction=(1 - frac),
             dropout_fraction=dropout_fraction,
             l2_reg=l2_reg,
             epochs=epochs)
        (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err,
         exch_mae, exch_rmse, exch_max_err, disp_mae, disp_rmse, disp_max_err,
         ind_mae, ind_rmse, ind_max_err) = sapt_net.sapt_errors(
             test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
             test_disp, disp_pred, test_ind, ind_pred)
        csv_file = []
        line = [
            "Sample from %s data ensemble, which had avg MAE %s Std Dev %s. This network has MAE %s."
            % (frac, avg_mae, std_mae, en_mae)
        ]
        csv_file.append(line)
        line = [
            'energy', ' energy_pred', ' elec', ' elec_pred', ' exch',
            ' exch_pred', ' ind', ' ind_pred', ' disp', ' disp_pred'
        ]
        csv_file.append(line)
        for i in range(len(energy_pred)):
            line = [
                test_en[i], energy_pred[i], test_elec[i], elec_pred[i],
                test_exch[i], exch_pred[i], test_ind[i], ind_pred[i],
                test_disp[i], disp_pred[i]
            ]
            csv_file.append(line)
        with open('%s_%s.csv' % (inputdir, frac), 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(csv_file)
        writeFile.close()

#fitting total E vs fitting components
if False:
    (test_en, energy_pred, avg_mae, std_mae) = sapt_net.sapt_net(
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
        val_split=val_split,
        data_fraction=(1 - val_split),
        dropout_fraction=dropout_fraction,
        l2_reg=l2_reg,
        epochs=epochs,
        batch_size=32,
        total_en_mode=True)
    csv_file = []
    line = [
        "Sample from %s data ensemble fitting TOTAL ENERGIES which had avg MAE %s Std Dev %s"
        % ((1 - val_split), avg_mae, std_mae)
    ]
    csv_file.append(line)
    line = ['energy', ' energy_pred']
    csv_file.append(line)
    for i in range(len(energy_pred)):
        line = [test_en[i], energy_pred[i]]
        csv_file.append(line)
    with open('%s_%s_total_fits.csv' % (inputdir, (1 - val_split)),
              'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()

    (test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
     test_disp, disp_pred, test_ind, ind_pred, avg_mae,
     std_mae) = sapt_net.sapt_net(
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
         val_split=val_split,
         data_fraction=(1 - val_split),
         dropout_fraction=dropout_fraction,
         l2_reg=l2_reg,
         epochs=epochs,
         batch_size=32)

    (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err, exch_mae,
     exch_rmse, exch_max_err, disp_mae, disp_rmse, disp_max_err,
     ind_mae, ind_rmse, ind_max_err) = sapt_net.sapt_errors(
         test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
         test_disp, disp_pred, test_ind, ind_pred)

    csv_file = []
    line = [
        "Sample from %s data ensemble fitting COMPONENT ENERGIES which had avg MAE %s Std Dev %s. This network has MAE %s."
        % ((1 - val_split), avg_mae, std_mae, en_mae)
    ]
    csv_file.append(line)
    line = ['energy', ' energy_pred']
    csv_file.append(line)
    for i in range(len(energy_pred)):
        line = [test_en[i], energy_pred[i]]
        csv_file.append(line)
    with open('%s_%s_component_fits.csv' % (inputdir, (1 - val_split)),
              'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()

#error by varying layers and nodes
if False:
    for layers in [2, 3, 4]:
        nodes = []
        for num in [100, 200, 300]:
            for i in range(layers):
                nodes.append(num)

            (test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred,
             test_disp, disp_pred, test_ind, ind_pred, avg_mae,
             std_mae) = sapt_net.sapt_net(
                 sym_input,
                 aname,
                 inputdir,
                 energy,
                 elec,
                 exch,
                 ind,
                 disp,
                 n_layer=layers,
                 nodes=nodes,
                 val_split=val_split,
                 data_fraction=(1 - val_split),
                 dropout_fraction=dropout_fraction,
                 l2_reg=l2_reg,
                 epochs=150,
                 batch_size=32)

            (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err,
             exch_mae, exch_rmse, exch_max_err, disp_mae, disp_rmse,
             disp_max_err, ind_mae,
             ind_rmse, ind_max_err) = sapt_net.sapt_errors(
                 test_en, energy_pred, test_elec, elec_pred, test_exch,
                 exch_pred, test_disp, disp_pred, test_ind, ind_pred)

            csv_file = []
            line = [
                "Sample from %s data ensemble fitting COMPONENT ENERGIES which had avg MAE %s Std Dev %s. This network has MAE %s."
                % ((1 - val_split), avg_mae, std_mae, en_mae)
            ]
            csv_file.append(line)
            line = [
                'energy', ' energy_pred', ' elec', ' elec_pred', ' exch',
                ' exch_pred', ' ind', ' ind_pred', ' disp', ' disp_pred'
            ]
            csv_file.append(line)
            for i in range(len(energy_pred)):
                line = [
                    test_en[i], energy_pred[i], test_elec[i], elec_pred[i],
                    test_exch[i], exch_pred[i], test_ind[i], ind_pred[i],
                    test_disp[i], disp_pred[i]
                ]
                csv_file.append(line)
            with open('%s_%s_%s_att2.csv' % (inputdir, num, layers),
                      'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(csv_file)
            writeFile.close()
            nodes = []
