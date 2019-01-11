from __future__ import print_function
import tensorflow_probability as tfp
import tensorflow as tf
import functools


class ElemBayesModel(object):
    """The ElemBayesModel class defines an inhereted neural network 
    definition returning only predictions since we usually can't 
    access labels for individual atomic properties

    Args:
        data (tf.float32 tensor): Symmetry functions for this atom
        grads (tf.float32 tensor): Gradient of loss function with
                                    respect to weights & biases in
                                    this network

    Attributes:
        data (tf.float32 tensor): Symmetry functions for this atom
        prediction (tf.float32): Atom energy contribution prediction
        
    """

    def __init__(self, inp_shape, atomic_number):
        self.atomic_number = atomic_number
        self.inp_shape = inp_shape
        self.target_size = 1
        #self.grads = grads
        #self.atom_type = atom_type
 
    def prediction(self):
        #data_size = int(self.data.get_shape()[1])
        print("\nSetting up NN for element %s"%self.atomic_number)
        prediction = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.inp_shape,)),
            tfp.layers.DenseFlipout(50, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(50, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(self.target_size)
        ]) 
        return prediction
