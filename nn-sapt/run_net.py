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



#train 1 net as specified
def standard_run(sym_input,aname,inputdir,geom_files,energy,elec,exch,
                ind,disp,n_layer,nodes,mt_vec,val_split,
                dropout_fraction,l2_reg,epochs, results_name):
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
    with open('%s_results.csv' % (inputdir), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_file)
    writeFile.close()
    """ 
    test_xyz_path = None
    data_obj = routines.DataSet(test_xyz_path,"../data/crystallographic_data/2019-01-05-CSD-NMA-Aniline-xyz-nrgs-outs/FSAPT0/SAPT0-NRGS-COMPONENTS.txt",None,"kcal/mol")
    (_aname,_atom_tensor,_xyz,testing_elec,testing_ind,testing_disp,testing_exch,
                testing_energy,geom_files) = data_obj.read_from_target_list()
    model_tests.evaluate_model(model, "NMA-Aniline-crystallographic", geom_files, results_name, testing_energy, testing_elec, testing_exch, testing_ind, testing_disp)
    K.clear_session()
    """

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    inputdirs = ["Aniline_10k_Training", "Indole_10k_Training", "MeOH_10k_Training", "NMe-acetamide_13k_Training"]
    xyz_paths = ["../data/NMA_Aniline_Step_4",
                "../data/NMA_Indole_Step_4",
                "../data/NMA_MeOH_Step_4",
                "../data/NMA_NMA_Step_4"]

    for i in range(len(inputdirs)):
        sym_input = []
        inputdir = inputdirs[i]
        xyz_path = xyz_paths[i]
        print("Collecting properties and geometries...\n")
        t1 = time.time()
        NNff = NNforce_field('GA_opt', 0, 0)
        #(aname, xyz, elec, exch, ind, disp, energy, atom_tensor, en_units) = routines.read_sapt_data(inputdir)
        #data_obj = routines.DataSet(
        #    "../data/Final_Sampling_Alex/%s_Step_4_AlexSampling.csv" % inputdir, "alex", "kcal/mol")
        data_obj = routines.DataSet(xyz_path,"../data/%s.csv"%inputdir,"non-alex","kcal/mol")    
        
        (aname, atom_tensor, xyz, elec, ind,
            disp, exch, energy, geom_files) = data_obj.read_from_target_list()
        
        #(testing_aname, testing_atom_tensor, testing_xyz, testing_elec,
        #     testing_ind, testing_disp, testing_exch, testing_energy,
        #     testing_geom_files, aname, atom_tensor, xyz, elec, ind, disp, exch,
        #     energy, geom_files) = data_obj.read_from_target_list()
        #for i in range(len(aname[0])):
        #    print(aname[0][i])
        
        t2 = time.time()
        elapsed = math.floor(t2 - t1)
        print("Properties and geometries collected in %s seconds\n" % elapsed)
        
        print("Loading symmetry functions from file...\n")
        t1 = time.time()
        sym_dir = "../data/%s_sym_inp" % inputdir
        #for i in range(
        #        len([
        #            name for name in os.listdir("%s" % sym_dir)
        #            if os.path.isfile(os.path.join(sym_dir, name))
        #        ])):
        for name in geom_files:
            sym_input.append(np.load("%s_symfun.npy"%os.path.join(sym_dir, name)))
        
        sym_input = routines.scale_symmetry_input(sym_input)
        
        t2 = time.time()
        elapsed = math.floor(t2 - t1)
        print("Symmetry functions loaded in %s seconds\n" % elapsed)
        
        dropout_fraction = 0.05
        l2_reg = 0.01
        nodes = [200, 200]
        val_split = 0.01
        epochs = 100
        n_layer = len(nodes)
        #mt_vec = [0.6, 0.1, 0.1, 0.1, 0.1]
        #results_name = "%s_tot_en_frac_%.1g"%(inputdir,float(mt_vec[0]))
        #standard_run(sym_input,aname,inputdir,geom_files,energy,elec,exch,
        #                ind,disp,n_layer,nodes,mt_vec,val_split,
        #                dropout_fraction,l2_reg,epochs,results_name)
        for j in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            mt_vec = [1 - j, j / 4, j / 4, j / 4, j / 4]
            results_name = "%s_tot_en_frac_%.1g"%(inputdir,float(mt_vec[0]))
            standard_run(sym_input,aname,results_name,geom_files,energy,
                         elec,exch,ind,disp,n_layer,nodes,mt_vec,val_split,   
                         dropout_fraction,l2_reg,epochs,results_name)
