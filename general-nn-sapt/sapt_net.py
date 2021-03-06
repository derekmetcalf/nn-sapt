from __future__ import print_function
import os
import csv
import sys
import math
import time
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, Dropout, Concatenate, dot, concatenate, add
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import regularizers
from sys import stdout
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import keras
import routines
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
from symfun_parameters import *
from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('add_2_mean_absolute_error'))
        self.val_losses.append(logs.get('val_add_2_mean_absolute_error'))
        self.i += 1
        
        #clear_output(wait=True)
        plt.ion()
        plt.show()
        plt.cla()
        if epoch < 5:
            plt.plot(self.x, self.losses, label="train MAE")
            plt.plot(self.x, self.val_losses, label="val MAE")
        elif epoch < 11:
            plt.plot(self.x[4:], self.losses[4:], label="train MAE")
            plt.plot(self.x[4:], self.val_losses[4:], label="val MAE")
        elif epoch < 31:
            plt.plot(self.x[10:], self.losses[10:], label="train MAE")
            plt.plot(self.x[10:], self.val_losses[10:], label="val MAE")
        else:
            plt.plot(self.x[30:], self.losses[30:], label="train MAE")
            plt.plot(self.x[30:], self.val_losses[30:], label="val MAE")

        plt.legend()
        plt.grid(alpha=0.2)
        plt.draw()
        plt.savefig("most_recent_train")
        plt.pause(0.001)

def sapt_net(sym_input,
             aname,
             results_name,
             energy,
             elec,
             exch,
             ind,
             disp,
             n_layer,
             nodes,
             multitarget_vec,
             val_split,
             data_fraction,
             dropout_fraction,
             l2_reg,
             epochs,
             geom_files,
             batch_size=32,
             simulations=1,
             folds=5,
             total_en_mode=False):
    """
    Construct and train NN-SAPT model with provided parameters.

    Keras allows for the construction of a different neural network model
    for each type of atom (C, H, O, etc.). The total output values (usually
    SAPT energies) are taken as a sum of atomic contributions and training
    is performed by backpropagating those errors through all networks 
    simultaneously.

    Reasonable flexibility is allowed just by calling this function 
    with options, but considerable architectural changes will require
    amendments to this function itself.

    """

    (ntype, atype, unique_list) = routines.create_atype_list(aname,
                                                routines.atomic_dictionary())
    #sym_input = np.transpose(sym_input, (1,0,2))
    y = np.transpose(np.array([energy, elec, exch, ind, disp]))
    val_scores = []
   
    max_atoms = 0
    for i in range(len(aname)):
        if len(aname[i]) > max_atoms:
            max_atoms = len(aname[i])
    print(sym_input.shape)
    NNff = NNforce_field('GA_opt', 0, 0)
    NNff2 = NNforce_field('GA_opt',0,0)
    NNunique = []
    
    geom_files_train, geom_files_test, X_train, X_test, y_train, y_test = train_test_split(
        geom_files, sym_input, y, test_size=val_split, shuffle=True, random_state=42)

    X_train = np.transpose(X_train, (1, 0, 2))
    X_test = np.transpose(X_test, (1, 0, 2))
    X_train = list(X_train)
    X_test = list(X_test)
    y_train = list(y_train.T)
    y_test = list(y_test.T)


    if total_en_mode == True:
        y_train = np.sum(y_train, axis=1)
        y_test = np.sum(y_test, axis=1)
    print("Setting up neural networks...\n")
    t1 = time.time()
    for i_type in range(ntype):
        NNelement = []
        i_size_l = nodes[0]
        for n in range(n_layer - 1):
            i_size_l = nodes[n]
            NN = Dense(
                i_size_l,
                activation='relu',
                use_bias=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f"{i_type}_{n}")
            NNelement.append(NN)
            NN = Dropout(dropout_fraction)
            NNelement.append(NN)
        i_size_l = 4
        if total_en_mode == True:
            i_size_l = 1
        NN = Dense(i_size_l, kernel_regularizer=regularizers.l2(l2_reg))
        NNelement.append(NN)
        NNunique.append(NNelement)
    
    NNtotal = []
    inputs = []
    molec_eval = []
    for i_atom in range(max_atoms):
        ind_net_outs = []
        typeNN = NNff.element_force_field[aname[0][0]]
        typeNN_intra = NNff2.element_force_field[aname[0][0]]
        i_size = len(typeNN.radial_symmetry_functions) + len(
            typeNN.angular_symmetry_functions) + len(
            typeNN_intra.radial_symmetry_functions) + len(
            typeNN_intra.angular_symmetry_functions)
        atom_input = Input(shape=(i_size+len(NNunique), ))
        sym_inp = Lambda(lambda x: K.slice(x,[0,0],[-1,i_size]),
                            output_shape=(i_size, ))(atom_input)
        atom_mask = Lambda(lambda x: K.slice(x,[0,i_size],[-1,-1]),
                            output_shape=(len(NNunique), ))(atom_input)


        for j_atomnet in range(len(NNunique)):
            for k_layer in range(len(NNunique[j_atomnet])):
                if k_layer == 0:
                    layer = NNunique[j_atomnet][k_layer](sym_inp)
                else:
                    if "Dropout" not in str(NNunique[j_atomnet][k_layer]):
                        layer = NNunique[j_atomnet][k_layer](layer)
                    else:
                        layer = NNunique[j_atomnet][k_layer](layer,training=True)
                        
            layer = Lambda(lambda x: K.expand_dims(x),
                            output_shape=(4,1, ))(layer)
            ind_net_outs.append(layer)
        ind_net_outs = concatenate(ind_net_outs, axis=-1)
        atom_outs = dot([ind_net_outs, atom_mask], axes=(-1,-1))
        molec_eval.append(atom_outs)
        inputs.append(atom_input)
    
    atom_component_preds = Lambda(lambda x: K.expand_dims(x), name="atom_comps")(molec_eval)
    component_predictions = add(molec_eval)
    elst_tensor = Lambda(
        lambda component_predictions: K.tf.gather(
            component_predictions, [0], axis=1),
        output_shape=(1, ))(component_predictions)
    exch_tensor = Lambda(
        lambda component_predictions: K.tf.gather(
            component_predictions, [1], axis=1),
        output_shape=(1, ))(component_predictions)
    ind_tensor = Lambda(
        lambda component_predictions: K.tf.gather(
            component_predictions, [2], axis=1),
        output_shape=(1, ))(component_predictions)
    disp_tensor = Lambda(
        lambda component_predictions: K.tf.gather(
            component_predictions, [3], axis=1),
        output_shape=(1, ))(component_predictions)
    total_predictions = add(
        [elst_tensor, exch_tensor, ind_tensor, disp_tensor])
    predictions = concatenate([total_predictions, component_predictions])

    model = Model(
        inputs=inputs,
        outputs=[
            total_predictions, elst_tensor, exch_tensor, ind_tensor,
            disp_tensor
        ])
    adam = keras.optimizers.Adam(lr=0.0005)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=['mae'],
        loss_weights=multitarget_vec)

    atom_model = Model(inputs=inputs, outputs=atom_component_preds)

    t2 = time.time()
    elapsed = math.floor(t2 - t1)
    print(
        "Neural network model created and compiled in %s seconds\n" % elapsed)

    print("Fitting neural network to property data...\n")
    t1 = time.time()
    
    plot_losses = PlotLosses()
    es = keras.callbacks.EarlyStopping(monitor='val_add_2_mean_absolute_error', mode='min',patience=100) 
    mc = keras.callbacks.ModelCheckpoint(f"{results_name}_model.h5", monitor="val_add_2_mean_absolute_error", save_best_only=True, verbose=1, mode="min")
    #tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test),epochs=epochs,verbose=2, callbacks=[es,mc])#callbacks=[tensorboard]
    t2 = time.time()
    elapsed = math.floor((t2 - t1) / 60.0)
    print("Neural network fit in %s minutes\n" % elapsed)

    model.summary()

    test_en = y_test[0]
    test_elec = y_test[1]
    test_exch = y_test[2]
    test_ind = y_test[3]
    test_disp = y_test[4]
    energy_pred = np.zeros(len(y_test))
    elec_pred = np.zeros(len(y_test))
    exch_pred = np.zeros(len(y_test))
    ind_pred = np.zeros(len(y_test))
    disp_pred = np.zeros(len(y_test))
    (energy_pred, elec_pred, exch_pred, ind_pred,
     disp_pred) = model.predict_on_batch(X_test)
    model.save("%s_model.h5" % results_name)
    atom_model.save("%s_atomic_model.h5" % results_name)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))
    val_scores.append(scores[1])

    print("%.2f (+/-) %.2f" % (np.mean(val_scores), np.std(val_scores)))
    return model, test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred, test_disp, disp_pred, test_ind, ind_pred, np.mean(
        val_scores), np.std(val_scores), atom_model, geom_files_train, geom_files_test


def sapt_errors(energy, energy_pred, elec, elec_pred, exch, exch_pred, disp,
                disp_pred, ind, ind_pred):
    """Compute standard error metrics for SAPT"""
    en_mae, en_rmse, en_max_err = routines.error_statistics(
        energy, energy_pred)
    elec_mae, elec_rmse, elec_max_err = routines.error_statistics(
        elec, elec_pred)
    exch_mae, exch_rmse, exch_max_err = routines.error_statistics(
        exch, exch_pred)
    disp_mae, disp_rmse, disp_max_err = routines.error_statistics(
        disp, disp_pred)
    ind_mae, ind_rmse, ind_max_err = routines.error_statistics(ind, ind_pred)

    return (en_mae, en_rmse, en_max_err, elec_mae, elec_rmse, elec_max_err,
            exch_mae, exch_rmse, exch_max_err, disp_mae, disp_rmse,
            disp_max_err, ind_mae, ind_rmse, ind_max_err)

