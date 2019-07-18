import os
import symfun
import datetime
import bpnn_handling
import numpy as np
import tensorflow as tf


def shuffle(xyz_l, mask_l, xyz_all, Z, labels):
    """ Shuffle data """
    perm = np.random.permutation(len(labels))
    perm = [[ind] for ind in perm]

    for i in range(len(xyz_l)):
        xyz_l[i] = tf.gather_nd(xyz_l[i], perm)
    for i in range(len(mask_l)):
        mask_l[i] = tf.gather_nd(mask_l[i], perm)
    xyz_all = tf.gather_nd(xyz_all, perm)
    Z = tf.gather_nd(Z, perm)
    labels = tf.gather_nd(labels, perm)

    return xyz_l, mask_l, xyz_all, Z, labels


def extract_val(xyz_l, mask_l, xyz_all, Z, labels):
    """ Splits data into training and validation """
    M_tot = len(labels)
    M_v = int(0.2 * M_tot)
    M_tr = M_tot - M_v

    inds_tr = [[ind] for ind in np.arange(0, M_tr)]
    inds_v = [[ind] for ind in np.arange(M_tr, M_tot)]

    xyz_l_v = []
    for i, ca in enumerate(xyz_l):
        xyz_l_v.append(tf.gather_nd(ca, inds_v))
        xyz_l[i] = tf.gather_nd(ca, inds_tr)
    mask_l_v = []
    for i, cm in enumerate(mask_l):
        mask_l_v.append(tf.gather_nd(cm, inds_v))
        mask_l[i] = tf.gather_nd(cm, inds_tr)
    xyz_all_v = tf.gather_nd(xyz_all, inds_v)
    xyz_all = tf.gather_nd(xyz_all, inds_tr)
    Z_v = tf.gather_nd(Z, inds_v)
    Z = tf.gather_nd(Z, inds_tr)
    labels_v = tf.gather_nd(labels, inds_v)
    labels = tf.gather_nd(labels, inds_tr)

    return xyz_l, xyz_l_v, mask_l, mask_l_v, xyz_all, xyz_all_v, Z, Z_v, labels, labels_v

def batch_inds(N, b):
    """ Helper function for batching """
    b_begin = [0]
    b_size = [b]
    while b_begin[-1] + b_size[-1] < N:
        b_begin.append(b_begin[-1] + b)
        b_size.append(min(b, N - b_begin[-1]))
    return b_begin, b_size


@tf.function
def make_features(bpnn, xyz_l, xyz_all, Z):
    """ Make rsyms from coordinates and nuclear charges """

    # calculate distance matrices for each atom type for each molecule
    dists = []
    for xyz in xyz_l:
        dists.append(symfun.get_distances_batch(xyz, xyz_all) )

    # calculate rsyms for each atom type for each molecule
    Xs = []
    for dist in dists:
        Xs.append(symfun.get_rsym_batch(Z, dist, bpnn.sym_params))

    return Xs

@tf.function
def predict_on_test_batch(bpnn, xyz_l, xyz_all, Z, mask_l, y):
    """ Perform inference on a batch of molecules """

    Xs = make_features(bpnn, xyz_l, xyz_all, Z)
    
    # make inference for each atom type
    pred = tf.zeros([Xs[0].shape[0]], dtype=tf.float64)
    for i in range(len(bpnn.models)):
        pred_atom = bpnn.models[i](Xs[i], training=False)  # M_b x A x 1
        pred_atom = tf.multiply(pred_atom[:, :, 0], mask_l[i])  # M_b x A
        pred_atom = tf.reduce_sum(pred_atom, 1)  # M_b
        pred += pred_atom

    # gradient of preduction w.r.t carteisan coords (forces)
    grads_xyz = tf.gradients(pred, xyz_l)

    loss_energy = bpnn.compute_loss(y, pred)
    loss_grad = tf.add_n([tf.reduce_sum(tf.norm(grad, axis=2)) for grad in grads_xyz])

    loss = loss_energy # + loss_grad

    return loss, pred, loss_grad

def predict_on_test(bpnn, xyz_l, xyz_all, Z, mask_l, y, b_begin, b_size):
    """ Perform inference on a batch of molecules """

    losses = []
    preds = []
    loss_grads = []

    for i in range(len(b_begin)):
        xyz_l_b = [
            tf.slice(x, [b_begin[i], 0, 0], [b_size[i], -1, -1])
            for x in xyz_l
        ]
        xyz_all_b = tf.slice(xyz_all, [b_begin[i], 0, 0],
                             [b_size[i], -1, -1])
        Z_b = tf.slice(Z, [b_begin[i], 0], [b_size[i], -1])

        mask_l_b = [
            tf.slice(m, [b_begin[i], 0], [b_size[i], -1]) for m in mask_l
        ]
        y_b = tf.slice(y, [b_begin[i]], [b_size[i]])

        loss, pred, loss_grad = predict_on_test_batch(bpnn, xyz_l_b, xyz_all_b, Z_b, mask_l_b, y_b)

        losses.append(loss)
        preds.append(pred)
        loss_grads.append(loss_grad)

    return tf.add_n(losses), tf.concat(preds, axis=0), tf.add_n(loss_grads)


@tf.function
def train_batch(bpnn,  xyz_l, xyz_all, Z, masks, y):
    """ Perform inference and a single backprop on a batch of molecules"""

    train_vars = []
    for model in bpnn.models:
        train_vars += model.trainable_variables

    # predict energies and compute loss
    
    dists = []
    for xyz in xyz_l:
        dists.append(symfun.get_distances_batch(xyz, xyz_all) )

    Xs = []
    for dist in dists:
        Xs.append(symfun.get_rsym_batch(Z, dist, bpnn.sym_params))

    pred = predict_on_train(bpnn, Xs, masks)
    grads_xyz = tf.gradients(pred, xyz_l)
    #tf.print(grads_xyz) # these gradients look OK

    loss_energy = bpnn.compute_loss(y, pred)
    loss_grads = tf.add_n([tf.reduce_sum(tf.norm(grad, axis=2)) for grad in grads_xyz])

    loss = loss_energy
    #loss = loss_grads
    #loss = loss_energy + loss_grads

    # update model variables
    grads_model = tf.gradients(loss, train_vars)
    #tf.print(grads_model) # these gradients are always 0.0
    bpnn.optimizer.apply_gradients(zip(grads_model, train_vars))
    
    return loss, pred


@tf.function
def predict_on_train(bpnn, Xs, masks):

    pred_total = tf.zeros([Xs[0].shape[0]], dtype=tf.float64)
    for i in range(len(bpnn.models)):
        pred = bpnn.models[i](Xs[i], training=True)
        pred = tf.multiply(pred[:, :, 0], masks[i])
        pred = tf.reduce_sum(pred, 1)
        pred_total += pred

    return pred_total


def train(bpnn, xyz_l, mask_l, xyz_all, Z, labels):
    """ Main BPNN training loop """

    # randomly shuffle the input (TODO: make less ugly)
    xyz_l, mask_l, xyz_all, Z, labels = shuffle(xyz_l, mask_l, xyz_all, Z,
                                                labels)
    # partition input into train/validation (TODO: make less ugly)
    xyz_l, xyz_l_v, mask_l, mask_l_v, xyz_all, xyz_all_v, Z, Z_v, labels, labels_v = extract_val(
        xyz_l, mask_l, xyz_all, Z, labels)

    A = len(bpnn.models)  # number of atom types
    E = bpnn.epochs  # number of epochs
    b = bpnn.batch_size  # batch size

    M = len(labels)  # number of training molecules
    b_begin, b_size = batch_inds(M, b)  # useful indices for batching
    n = int(np.ceil(M / b))  # number of batches
    a = int(tf.add_n([tf.reduce_sum(m) for m in mask_l])) # number of validation atoms

    M_v = len(labels_v)  # number of validation molecules
    b_begin_v, b_size_v = batch_inds(M_v, b)  # useful indices for batching
    n_v = int(np.ceil(M_v / b))  # number of batches
    a_v = int(tf.add_n([tf.reduce_sum(m) for m in mask_l_v])) # number of validation atoms

    print(f'There are {A} unique atom types in this BPNN')
    print(f'Train / Validation split of ({M},{M_v}) molecules or ({a},{a_v}) atoms')
    print(f'Training for {E} epochs, batch size of {b} ({n} batches per epoch)')

    # keep track of the best validation MAE
    best_MAE_v = 1.0e9

    for e in range(E):

        # shuffle training data each epoch
        xyz_l, mask_l, xyz_all, Z, labels = shuffle(xyz_l, mask_l, xyz_all, Z,
                                                    labels)

        epoch_loss = 0.0
        epoch_ae = 0.0

        # train a batch at a time
        for i in range(n):
            xyz_l_b = [
                tf.slice(x, [b_begin[i], 0, 0], [b_size[i], -1, -1])
                for x in xyz_l
            ]
            xyz_all_b = tf.slice(xyz_all, [b_begin[i], 0, 0],
                                 [b_size[i], -1, -1])
            mask_l_b = [
                tf.slice(m, [b_begin[i], 0], [b_size[i], -1]) for m in mask_l
            ]
            Z_b = tf.slice(Z, [b_begin[i], 0], [b_size[i], -1])
            labels_b = tf.slice(labels, [b_begin[i]], [b_size[i]])
            batch_loss, pred = train_batch(bpnn, xyz_l_b,
                                           xyz_all_b, Z_b, mask_l_b, labels_b)
            epoch_loss += batch_loss.numpy() * b_size[i]
            epoch_ae += np.sum(np.absolute((labels_b - pred).numpy()))

        # Training Error Statistics
        RMSE = np.sqrt(epoch_loss / M)
        MAE = epoch_ae / M

        # Validation Error Statistics
        loss_v, pred_v, lg_v, = predict_on_test(bpnn, xyz_l_v, xyz_all_v, Z_v,
                                         mask_l_v, labels_v, b_begin_v, b_size_v)

        RMSE_v = np.sqrt(loss_v.numpy())
        MAE_v = np.average(np.absolute((labels_v - pred_v).numpy()))

        found_new_min = '*' if MAE_v < best_MAE_v else ''
        best_MAE_v = min(MAE_v, best_MAE_v)

        print(
            f'Epoch: {e:3d}  ||  RMSE: {RMSE:6.2f} / {RMSE_v:6.2f}  ||  MAE: {MAE:6.2f} / {MAE_v:6.2f} {found_new_min}'
        )
        print(
                f'            ||  Mean force per atom (kcal/mol/Ang) : {lg_v/a_v:6.2f}'
        )
    return bpnn.models


def save_models(bpnn, model_loc=None):
    #Create directory at loc or with standard naming scheme
    if not os.path.isdir("models"):
        os.mkdir("models")
    if model_loc:
        if os.path.isdir(f"models/{model_loc}"):
            num = 0
            model_loc = f"model_{str(num).zfill(5)}"
            while os.path.isdir(f"models/{model_loc}"):
                num += 1
                model_loc = f"model_{str(num).zfill(5)}"
            os.mkdir(f"./models/{model_loc}")
            print(f"Save path already exists -- override save at models/{model_loc}") 
    elif not model_loc or os.path.isdir(f"models/{model_loc}"):
        num = 0
        model_loc = f"model_{str(num).zfill(5)}"
        while os.path.isdir(f"models/{model_loc}"):
            num += 1
            model_loc = f"model_{str(num).zfill(5)}"
        os.mkdir(f"./models/{model_loc}")
        print(f"Models saved at models/{model_loc}")

    else:
        os.mkdir(f"models/{model_loc}")
        print(f"Models saved at models/{model_loc}")

    #Collect and save models at save path
    atom_dict = {   1 : "H",
                    6 : "C",
                    7 : "N",
                    8 : "O",
                    9 : "F",
                    16: "S"}
    for i, model in enumerate(bpnn.models):
        atomtype = atom_dict[bpnn.train_atoms[i]]
        tf.saved_model.save(model, f"./models/{model_loc}/{atomtype}") 

    #Store important info about model @ {model_loc}/info.txt
    with open(f"./models/{model_loc}/info.txt", "w+") as info:
        info.write(f"finished training at {datetime.datetime.now()}\n")
        info.write(f"----------------------------------------------\n")
        info.write(f"Atomic linear regression parameters:\n")
        for i, coeff in enumerate(bpnn.coeffs):
            info.write(f"   {atom_dict[bpnn.train_atoms[i]]} {coeff}\n")
        info.write(f"Network nodes per layer: {bpnn.nodes}\n")
        info.write(f"Epochs                 : {bpnn.epochs}\n")
        info.write(f"Activation functions   : {bpnn.activation}\n")
        info.write(f"Symfun parameters      :\n{bpnn.sym_params.numpy()}\n")
    
    return
