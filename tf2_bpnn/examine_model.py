import os
import bpnn
import train
import symfun
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    # get input / output tensors from file
    xyz_l, mask_l, xyz_all, Z, labels = symfun.get_qm9(100)

    # one simple NN model per atom type
    A = len(xyz_l)
    models = [train.atom_net() for i in range(A)]

    l = bpnn.labels.numpy()
    print(f'Average Molecule Energy is {np.average(l):.2f} +- {np.std(l):.2f}')

    # train the model
    #train.train(models, optimizer, xyz_l, mask_l, xyz_all, Z, labels)
