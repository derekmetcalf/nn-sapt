import train
import symfun
import bpnn_handling 
import tensorflow as tf

# specify model save location
model_loc = "./models/model_00015"

# get input for testing
xyz_l, mask_l, xyz_all, Z, labels, test_atom_Z, coeffs = symfun.get_qm9(1000)

# load train-time symfun params
train_params = bpnn_handling.load_params(model_loc)

# load train-time atoms and regression coefficients
train_atoms, train_coeffs = bpnn_handling.load_coeffs(model_loc)
train_atom_Z = symfun.get_Z(train_atoms)

# ensure train and test data contain same atomtypes & ordering
bpnn_handling.train_test_check(train_atom_Z, test_atom_Z)

# initialize stupid bpnn arguments
# BPNN class ought to just have these as default arguments
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
epochs = 0
batch_size = 32
nodes = []
compute_loss = tf.keras.losses.MeanSquaredError()

# instantiate testing BPNN object
bpnn = bpnn_handling.BPNN(optimizer, batch_size, epochs, nodes, test_atom_Z, train_coeffs, train_params, compute_loss)

#load atom models
bpnn.load_models(model_loc, train_atoms)

# Models are loaded... the below doesn't work because I don't know how 
# the predict_on_test batching works, but all else seems good


err, preds, grads = train.predict_on_test(bpnn, xyz_l, xyz_all, Z, mask_l, labels, [], labels)
#print(err)
#print(preds)
#print(grads)
