from __future__ import print_function
import os
import sys
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, concatenate, add
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from bayes_model import ElemBayesModel
import routines3 as routines
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import pandas as pd
import FFenergy_openMM as saptff
from force_field_parameters import *

os.environ["CUDA_VISIBLE_DEVICES"]=""
inputfile='h2o_h2o_dummy_Eint_10mh.bohr'
pdb = PDBFile('h2o_template.pdb')

def build_input(inputfile, pdb, val_split): 
    # read input data, output as numpy arrays 
    (aname, xyz, energy ) = routines.read_sapt_data2( inputfile )
    (energy, ffenergy, residual) = saptff.resid( inputfile, pdb )
    # load the Neural network force field definitions
    NNff = NNforce_field( 'FF1',0,0 )

    # compute displacement tensor for all data points, output as numpy array
    rij,dr = routines.compute_displacements(xyz)

    # construct symmetry functions for NN input 
    sym_input = routines.construct_symmetry_input(NNff, rij, aname, ffenergy, val_split)
    (ntype, atype, atypename, atomic_number) = routines.create_atype_list(
        aname,routines.atomic_dictionary())
    print("\n\nDataset Characteristics:\n")
    print("aname: Atoms in the molecules\n%s\n"%(aname))
    print("ntype: Number of atomtypes in data\n%s\n"%(ntype))
    print("atype: Atoms in molecules NN indeces\n%s\n"%(atype))
    print("atypename: Atomtype names in data\n%s\n"%(atypename))
    print("atomic_number: Atomtype atomic numbers in data\n%s\n"%(atomic_number))
    #X1 = pd.DataFrame(sym_input[0])
    #X2 = pd.DataFrame(sym_input[1])
    #X = np.concatenate(sym_input[0],sym_input[1])
    X = sym_input
    y = residual
    print("Shape of feature input: %s,%s\n"%(len(X),X[0].shape))
    print("Shape of label input: %s\n"%(y.shape))
    if hasattr(X[0],'shape'):
        split_at = int(int(X[0].shape[0]) * (1. - val_split))
    else:
        split_at = int(int(X[0].shape[0]) * (1. - val_split))

    print("Validation split index: %s\n"%(split_at))
    X_train, X_test = (routines.slice_arrays(X, 0, split_at),
                routines.slice_arrays(X, split_at))
    y_train = y[0:split_at]
    y_test = y[(split_at):(len(y)+1)]
    #y_train, y_test = (routines.slice_arrays(y, 0, split_at),
    #            routines.slice_arrays(y, split_at))
    X_train = np.transpose(np.asarray(X_train),axes=(1,2,0))
    X_test = np.transpose(np.asarray(X_test),axes=(1,2,0))
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print("Training feature size: (%s,%s,%s)\n"%(X_train.shape))
    print("Test feature size: (%s,%s,%s)\n"%(X_test.shape))
    print("Training label size: %s\n"%(y_train.shape))
    print("Test label size: %s\n"%(y_test.shape))
    print("Training labels: %s\n"%(y_train))
    #X_train = tf.convert_to_tensor(X_train)
    #X_test = tf.convert_to_tensor(X_test) 
    #y_train = tf.convert_to_tensor(y_train)
    #y_test = tf.convert_to_tensor(y_test) 
    
    batch_size = 1
    heldout_size = batch_size
   
    #print(X_train[0].shape)
    #print(y_train)

    training_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    heldout_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    heldout_frozen = (heldout_dataset.take(heldout_size).repeat().batch(heldout_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()
    print(training_batches.output_types)
    print(training_batches.output_shapes)
    handle = tf.placeholder(tf.string,[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    sym_input, residual = feedable_iterator.get_next()
     
    return (sym_input,ntype,atype,atypename,residual,atomic_number,
                    aname,atype,NNff,training_iterator,heldout_iterator,handle)

def main(argv):
    del argv
    with tf.Graph().as_default():
        (sym_input,ntype,atype,atypename,residual
            ,atomic_number,aname,atype,NNff
                ,training_iterator,heldout_iterator,handle) = build_input(inputfile,pdb,0.2)
        with tf.name_scope("bayesian_neural_nets", values=[sym_input]):
            element_nets = []
            for element in range(len(atomic_number)):
                #with tf.name_scope("BNN_%s"%element, values=[sym_input]):
                network = ElemBayesModel(len(NNff.element_force_field[
                    aname[element]].symmetry_functions)+1,atomic_number[element])
                prediction = network.prediction()
                element_nets.append(prediction)
            atom_energy = []
            inputs=[]
            for i_atom in range(len(aname)):
                elem_sym_fun = NNff.element_force_field[aname[i_atom]]
                i_size = len(elem_sym_fun.symmetry_functions)
                atom_input = tf.keras.Input(shape=(i_size,))
                itype = atype[i_atom]
                atom_energy.append(element_nets[atype[i_atom]].output)
                inputs.append(atom_input)
            pred_en = tf.keras.layers.add(atom_energy)
            labels_distribution = tf.distributions.Normal(loc=pred_en,scale=pred_en/10)
            residual = tf.cast(residual,dtype=tf.float32) 
        neg_log_likelihood = -labels_distribution.log_prob(residual)
        kl = tf.reduce_sum(residual - pred_en) / tf.cast(tf.size(residual),tf.float32)
        elbo_loss = neg_log_likelihood + kl
        predictions = pred_en
        accuracy, accuracy_update_op = tf.metrics.accuracy(
            labels=residual, predictions=predictions)

        names = []
        qmeans = []
        qstds = []
        for i_atom in range(len(atypename)):
            for i,layer in enumerate(element_nets[i_atom].layers):
                q = layer.kernel_posterior
                names.append('Layer {}'.format(i))
                qmeans.append(q.mean())
                qstds.append(q.stddev())

        with tf.name_scope("train"):
            print("Initializing variables for training...")
            opt = tf.train.AdamOptimizer()

            train_op = opt.minimize(elbo_loss)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            train_handle = sess.run(training_iterator.string_handle())
            heldout_handle = sess.run(heldout_iterator.string_handle())
            print("Beginning training procedures")
            print(train_handle)
            for step in range(1000):
                _ = sess.run([train_op, accuracy_update_op],
                            feed_dict={handle: train_handle})
                print("Session loaded")
                if step % 100 == 0:
                    loss_value, accuracy_value = sess.run(
                        [elbo_loss, accuracy], feed_dict={handle: train_handle})
                    print("Step: {>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                        step, loss_value, accuracy_value))
                if (step+1) % 400 == 0:
                    probs = np.asarray([sess.run((labels_distribution.probs),
                                        feed_dict={handle: train_handle})
                                        for _ in range(50)])
                    mean_probs = np.mean(probs, axis=0)
                    
                    #image_vals, label_vals = sess.run((images, labels),
                    #                            feed_dict={handle: heldout_handle})
                    #heldout_lp = np.mean(np.log(mean_probs[np.range(mean_probs.shape[0]),
                    #                            label_vals.flatten()]))
                    #print(" ... Held-out nats: {:.3f}".format(heldout_lp))

                    qm_vals, qs_vals = sess.run((qmeans, qstds))
            
    return


if __name__ == "__main__":
    tf.app.run()
