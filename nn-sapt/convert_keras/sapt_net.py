from __future__ import print_function
import os
import csv
import sys
import math
import time
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, Dropout, concatenate, add
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import regularizers
from sys import stdout
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import routines
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
from symfun_parameters import *


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
             batch_size=32,
             simulations=1,
             folds=5,
             total_en_mode=False):
    """Construct and train NN-SAPT model with provided parameters.

    Keras allows for the construction of a different neural network model
    for each type of atom (C, H, O, etc.). The total output values (usually
    SAPT energies) are taken as a sum of atomic contributions and training
    is performed by backpropagating those errors through all networks 
    simultaneously.

    Reasonable flexibility is allowed just by calling this function 
    with options, but considerable architectural changes will require
    amendments to this function itself."""

    tf.enable_eager_execution()

    (ntype, atype) = routines.create_atype_list(aname,
                                                routines.atomic_dictionary())
    #sym_input = np.transpose(sym_input, (1,0,2))
    y = np.transpose(np.array([energy, elec, exch, ind, disp]))
    val_scores = []
   
 
    """
    kfold = KFold(folds)



    
    frac_after_fold = (1.0 - 1.0/float(folds))
    test_size = 1 - data_fraction/frac_after_fold

    if test_size<=1 and test_size>0:
        sym_input, xtrash, y, ytrash = train_test_split(sym_input,y,test_size=test_size)
    
    for train, test in kfold.split(sym_input): 
        X_train, X_test = sym_input[train], sym_input[test]
        y_train, y_test = y[train], y[test]
    
"""

    NNff = NNforce_field('GA_opt', 0, 0)
    NNunique = []
    inputelement = []
    X_train, X_test, y_train, y_test = train_test_split(
        sym_input, y, test_size=val_split)
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

    for i_atom in range(len(aname[0])):
        itype = atype[0][i_atom]
        typeNN = NNff.element_force_field[aname[0][i_atom]]
        print(typeNN.radial_symmetry_functions)
        i_size = len(typeNN.radial_symmetry_functions) + len(
            typeNN.angular_symmetry_functions)
        #atom_input = Input(shape=(i_size, ))

        NNatom = []
        for i_layer in range(len(NNunique[0])):
            if i_layer == 0:
                layer = NNunique[itype][i_layer](atom_input)
            else:
                if "Dropout" not in str(NNunique[itype][i_layer]):
                    layer = NNunique[itype][i_layer](layer)
                else:
                    layer = NNunique[itype][i_layer](layer)  #, training=True)
        inputs.append(atom_input)
        NNtotal.append(layer)
   

    #molec_input = Input(shape=(max_atoms,i_size))

    component_predictions = add(NNtotal)
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
    model.compile(
        optimizer='adam',
        loss="mean_squared_error",
        metrics=['mae'],
        loss_weights=multitarget_vec)
    t2 = time.time()
    elapsed = math.floor(t2 - t1)
    print(
        "Neural network model created and compiled in %s seconds\n" % elapsed)

    print("Fitting neural network to property data...\n")
    t1 = time.time()
    
    tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        callbacks=[tensorboard], epochs=epochs)
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
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))
    val_scores.append(scores[1])

    print("%.2f (+/-) %.2f" % (np.mean(val_scores), np.std(val_scores)))
    return model, test_en, energy_pred, test_elec, elec_pred, test_exch, exch_pred, test_disp, disp_pred, test_ind, ind_pred, np.mean(
        val_scores), np.std(val_scores)


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


"""
lin_elec = elec
lin_exch = exch
lin_ind = ind
lin_disp = disp
lin_energy = energy

f, axarr = plt.subplots(2,2)
axarr[0,0].scatter(elec,elec_pred,s=0.9,color='xkcd:red')
axarr[0,0].set_title('Electrostatics')
axarr[0,0].plot(elec,lin_elec,color='xkcd:coral')
axarr[0,1].scatter(exch,exch_pred,s=0.9,color='xkcd:green')
axarr[0,1].set_title('Exchange')
axarr[0,1].plot(exch,lin_exch,color='xkcd:coral')
axarr[1,0].scatter(ind,ind_pred,s=0.9,color='xkcd:blue')
axarr[1,0].set_title('Induction')
axarr[1,0].plot(ind,lin_ind,color='xkcd:coral')
axarr[1,1].scatter(disp,disp_pred,s=0.9,color='xkcd:orange')
axarr[1,1].set_title('Disperson')
axarr[1,1].plot(disp,lin_disp,color='xkcd:coral')


for ax in axarr.flat:
    ax.set(xlabel=('SAPT energies %s'%en_units), ylabel=('NN energies %s'%en_units))
plt.subplots_adjust(hspace=0.4)
plt.show()

plt.scatter(energy,energy_pred,s=0.9,color='xkcd:black')
plt.title('Interaction Energy')
plt.plot(energy,lin_energy,color='xkcd:coral')
plt.show()
"""
