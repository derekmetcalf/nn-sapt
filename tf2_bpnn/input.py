import train
import symfun
import bpnn_handling
import numpy as np
import tensorflow as tf

batch_size = 32
epochs = 3
nodes = [100, 100, 100]

# optimizer for updating NN params
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# loss function to optimize
compute_loss = tf.keras.losses.MeanSquaredError()

# get input / output tensors from file
xyz_l, mask_l, xyz_all, Z, labels, atom_Z, coeffs = symfun.get_qm9(1000)
sym_params = symfun.get_params()

# instantiate BPNN object
bpnn = bpnn_handling.BPNN(optimizer, batch_size, epochs, nodes, 
                    atom_Z, coeffs, sym_params, compute_loss)

# one simple NN model per atom type
bpnn.build_atom_nets()

# tensorify parameters
bpnn.prep_sym_params()

# train network
l = labels.numpy()
print(f'Average Molecule Energy is {np.average(l):.2f} +- {np.std(l):.2f}')
train.train(bpnn, xyz_l, mask_l, xyz_all, Z, labels)
 
train.save_models(bpnn)
