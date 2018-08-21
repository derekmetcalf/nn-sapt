from __future__ import print_function
import os
import sys
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Activation, Dropout, concatenate, add
from keras.utils import plot_model
from keras import regularizers
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import routines3 as routines
import keras.backend as K
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import symmetry_functions as sym
import FFenergy_openMM as saptff
from force_field_parameters import *

os.environ["CUDA_VISIBLE_DEVICES"]=""
inputfile='h2o_h2o_dummy_Eint_10mh.bohr'
pdb = PDBFile('h2o_template.pdb')

# read input data, output as numpy arrays 
(aname, xyz, energy ) = routines.read_sapt_data2( inputfile )
(energy, ffenergy, residual) = saptff.resid( inputfile, pdb )

# load the Neural network force field definitions
NNff = NNforce_field( 'FF1',0,0 )

# compute displacement tensor for all data points, output as numpy array
rij,dr = routines.compute_displacements(xyz)

# construct symmetry functions for NN input 
sym_input = routines.construct_symmetry_input( NNff , rij , aname , ffenergy,val_split=0 )
#*****************************************
# call this routine if we want to compute histrograms of symmetry input
# over the dataset.  This will print histograms and exit code
#******************************************
#flag = routines.test_symmetry_input(sym_input)

# the number of unique neural networks is the number of unique elements, create
# list linking elements to unique atomtypes
(ntype, atype, atypename, atomic_number) = routines.create_atype_list(
                                    aname,routines.atomic_dictionary())


# now construct NN model
# for keras 2.2, the Network/Model class is found in source keras/engine/network.py
# for keras 2.1.6, there is no Model class, but the highest class is 'layer', found in keras/engine/topology.py
# the class "Dense" is the standard type of neural-network layer, found in /keras/layers/core.py


#******* hidden layers *********
# outputs the atomic energies, so output dimension (units) is Natom
# use arctan for activation function

# here we initialize a list of element specific NNs, where each NN is a 1-D layer
# we need a unique layer object for each element, for each layer.  Thus loop over layers and elements

n_layer=3
NNunique=[]
inputelement=[]
dropout_fraction = 0.1
l2_reg = 0.01
# loop over elements, each final layer returns atomic energy of size '1'
for i_type in range(ntype):
    NNelement=[]
    i_size = len(NNff.element_force_field[atypename[i_type]].symmetry_functions) + 1
    # if last hidden layer, output size is 1, else output size
    # is same as input size
    i_size_l = i_size

    # create unique NN layer object for this element and this layer, don't
    # provide input tensor, as we want to keep this abstract object to apply
    # latter to individual atoms: Suppplying input would turn NN into tensor
    NN = Dense(i_size_l, activation='relu', 
                use_bias=True, kernel_regularizer=regularizers.l2(l2_reg))
    NNelement.append(NN)
    NN = Dropout(dropout_fraction)
    NNelement.append(NN)
    
    NN = Dense(i_size_l, activation='relu',
                use_bias=True, kernel_regularizer=regularizers.l2(l2_reg))
    NNelement.append(NN)
    NN = Dropout(dropout_fraction)
    NNelement.append(NN)
    
    NN = Dense(i_size_l, activation='relu',
                use_bias=True, kernel_regularizer=regularizers.l2(l2_reg))
    NNelement.append(NN)
    NN = Dropout(dropout_fraction)
    NNelement.append(NN)

    i_size_l = 1
    NN = Dense(i_size_l, kernel_regularizer=regularizers.l2(l2_reg))
    NNelement.append(NN)
    # now append layers
    NNunique.append(NNelement)

# now we have setup all unique NN layers, and we apply them to all atoms in the system
# to create connected tensors 

# note, while NNunique is list of layer objects,
# NNatom_layers is list of layer tensors
NNtotal=[]
inputs=[]
# now apply element specific layer to each atom
for i_atom in range(len(aname)):
    itype = atype[i_atom]
    # input size depends on element, as there could be different number of symmetry functions for each element
    typeNN = NNff.element_force_field[aname[i_atom]]
    i_size = len(typeNN.symmetry_functions) + 1
    atom_input = Input(shape=(i_size,))

    NNatom=[]
    # loop over layers
    for i_layer in range(len(NNunique[0])):
        print("Layer %s initiating" %i_layer)
        # here we turn object into tensor by supplying input tensor
        # if first hidden layer, supply 'atom_input' tensor
        if i_layer == 0 :
            layer = NNunique[itype][i_layer](atom_input)
        else :
            # layer on top of previous layer
            if "Dropout" not in str(NNunique[itype][i_layer]):
                layer = NNunique[itype][i_layer](layer)
            else:
                layer = NNunique[itype][i_layer](layer, training=True)
    # append input tensor, and total, layered NN tensor for this atom
    inputs.append(atom_input)
    NNtotal.append(layer)

# now concatenate the atom vectors
NNwhole = concatenate(NNtotal, axis=1)
predictions = add(NNtotal)

#******* output layer *********
# output system energy as sum of atomic energies, so output dimension (units) is 1,
# we are just summing atomic contributions here... K is a keras object set to either tensorflow or theanos


model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


print( "fitting Neural network to interaction energy data...")
history = model.fit(sym_input, residual, batch_size=4, epochs=150, validation_split=0.1)

#model.save('../models/basic1_model.h5')
#model.save_weights('../models/basic1_weights.h5')

print("done fitting Neural network")
 
model.summary()

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mae')
plt.ylabel('mae (millihartrees)')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#print( "predictions" )
y = model.predict_on_batch(sym_input)
liny = residual

model.save("model_test.h5")

y_mean = np.average(y)
SStot = 0
SSreg = 0
SSres = 0
for i in range(len(y)):
    SStot = SStot + (y[i] - y_mean)**2
    SSreg = SSreg + (liny[i] - y_mean)**2
    SSres = SSres + (y[i] - liny[i])**2
R2 = 1 - SSres/SStot
plt.scatter(residual,y)
plt.plot(residual,liny,color='xkcd:coral')
plt.title('residual comparison')
plt.ylabel('NN - FF Residual (kJ/mol)')
plt.xlabel('SAPT - FF Residual (kJ/mol)')
plt.text(-5,10,'R2 = %s'%(R2))
plt.show()
#for i in range(len(energy)):
#    print( energy[i] , y[i] )

line = np.linspace(-30,30)

fitEn = np.zeros(len(energy))
for i in range(len(energy)):
    fitEn[i] = y[i] + ffenergy[i]


plt.subplot(2,1,1)
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.plot(line,line,color='xkcd:black')
plt.scatter(energy, fitEn, s=0.9, color='xkcd:red')
plt.ylabel('NN + FF energy (kJ/mol)')


plt.subplot(2,1,2)
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.plot(line,line,color='xkcd:black')
plt.scatter(ffenergy, energy, s=0.9, color='xkcd:red')
plt.xlabel('SAPT energy (kJ/mol)')
plt.ylabel('FF energy (kJ/mol)')
plt.show()
