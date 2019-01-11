from __future__ import print_function
import os
import csv
import sys
import time
import math
import pickle
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, Dropout, concatenate, add
from keras.utils import plot_model
from keras import regularizers
from sys import stdout
import routines
import keras.backend as K
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
import symfun_parameters

"""Generate symmetry function environmental input descriptors.

"Runtime-maintained^TM" file to create symmetry functions of full datasets
for the purposes of training. Carefully utilize the DataSet object to get
inputs for best results.

"""

os.environ["CUDA_VISIBLE_DEVICES"] = ""
#inputdir="NMe-acetamide_Indole"
inputdirs = ["NMA-MeOH-crystallographic"]
#inputdirs = ["NMe-acetamide_Indole", "NMe-acetamide_Don--1H23triazole", "NMe-acetamide_Don--Benzene", "NMe-acetamide_DonH2--134oxadiazole"]

for inputdir in inputdirs:
    print("Scraping property, geometry data...\n")
    t1 = time.time()
    #(aname, xyz, energy, atom_tensor, en_units) = routines.read_QM9_data(inputdir)

    #data_obj = routines.DataSet("./2019-01-02-11-standard-donors-xyz-nrgs/SAPT0-NRGS-COMPONENTS/NMe-acetamide_Indole-SAPT0-NRGS-COMPONENTS.txt",None,"kcal/mol")
    data_obj = routines.DataSet(
        "./crystallographic_data/2018-12-26-CSD-xyznma-meoh-159/FSAPT0/SAPT0-NRGS-COMPONENTS.txt",
        None, "kcal/mol")

    if data_obj.mode == "alex":
        print("alex mode engaged")
        (test_aname, test_atom_tensor, test_xyz, test_elec, test_ind,
         test_disp, test_exch, test_energy, test_geom_files, train_aname,
         train_atom_tensor, train_xyz, train_elec, train_ind, train_disp,
         train_exch, train_energy,
         train_geom_files) = data_obj.read_from_target_list()

    else:
        (aname, atom_tensor, xyz, elec, ind, disp, exch, energy,
         geom_files) = data_obj.read_from_target_list()
    #(aname, xyz, elec, exch, ind, disp, energy, atom_tensor, en_units) = routines.read_sapt_data(inputdir)
    t2 = time.time()
    elapsed = math.trunc(t2 - t1)
    print("Property and geometry data scraped in %s seconds\n" % elapsed)
    NNff = symfun_parameters.NNforce_field('GA_opt', 0, 0)

    if data_obj.mode == "alex":
        train_path = "./%s_train_spatial_info" % inputdir
        test_path = "./%s_test_spatial_info" % inputdir
    else:
        path = "./%s_spatial_info" % inputdir

    def compute_spatial_info(xyz, path):
        print("Computing interatomic distances and angles...\n")
        t1 = time.time()
        if not os.path.isdir(path):
            os.mkdir(path)
        num_systems = len(xyz)
        routines.compute_displacements(xyz, path)
        routines.compute_thetas(num_systems, path)
        t2 = time.time()
        elapsed = math.trunc((t2 - t1) / 60.0)
        print("%s minutes spent computing interatomic distances and angles\n" %
              elapsed)
        return

    atom_dict = routines.atomic_dictionary()

    if data_obj.mode == "alex":
        compute_spatial_info(train_xyz, train_path)
        train_atomic_num_tensor = routines.get_atomic_num_tensor(
            train_atom_tensor, atom_dict)
        compute_spatial_info(test_xyz, test_path)
        test_atomic_num_tensor = routines.get_atomic_num_tensor(
            test_atom_tensor, atom_dict)
        train_xyz = []
        test_xyz = []
    else:
        compute_spatial_info(xyz, path)
        atomic_num_tensor = routines.get_atomic_num_tensor(
            atom_tensor, atom_dict)
        xyz = []
    print(
        "Constructing symmetry functions from distances, angles, atomic numbers, and hyperparameters...\n"
    )
    t1 = time.time()
    if data_obj.mode == "alex":
        routines.construct_symmetry_input(
            NNff,
            train_path,
            train_atom_tensor,
            np.zeros(len(train_energy)),
            train_atomic_num_tensor,
            val_split=0)
        routines.construct_symmetry_input(
            NNff,
            test_path,
            test_atom_tensor,
            np.zeros(len(test_energy)),
            test_atomic_num_tensor,
            val_split=0)
    else:
        routines.construct_symmetry_input(
            NNff,
            path,
            atom_tensor,
            np.zeros(len(energy)),
            atomic_num_tensor,
            val_split=0)

    t2 = time.time()
    elapsed = (t2 - t1) / 60
    print(
        "Done! %s minutes spent constructing symmetry functions.\n" % elapsed)
