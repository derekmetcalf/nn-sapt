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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    NNff = NNforce_field('GA_opt',0,0)
    
    #path = "../data/NMe-acetamide_Don--Aniline_crystallographic"
    path = "./NMA-Aniline_Step_3"
     
    (filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)
   
    sym_input = [] 
    for i in range(len(filenames)):
        file = f"{path}/{filenames[i]}_symfun.npy"
        sym_input.append(np.load(file)) 
    
    routines.test_symmetry_input(sym_input)
    #model = load_model("./NMA-Aniline_Step_3_model.h5")
     
    #results_name = "NMA_Aniline_Fake_data_testing"
    
    #evaluate_model(model, sym_input, filenames, results_name, tot_en,
    #                    elst, exch, ind, disp) 
