import routines
import time
import numpy as np
import tensorflow as tf
import symmetry_functions as sym
import FFenergy_openMM as saptff
import symfun_parameters
from tensorflow.python import debug as tf_debug

def O_net():
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="O_0",
                                kernel_initializer=weight_init[0],
                                bias_initializer=bias_init)
    layer = tf.nn.tanh(layer)
    for j in range(depth-1):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="O_"+str(j+1),
                        kernel_initializer=weight_init[j+1],
                        bias_initializer=bias_init)
        layer = tf.nn.tanh(layer)
        layer = tf.layers.dropout(layer, rate=0.05)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="O_last",
                            kernel_initializer=weight_init[depth],
                            bias_initializer=bias_init)
    O_out = layer
    return O_out

def C_net():
    
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="C_0",
                                kernel_initializer=weight_init[0],
                                bias_initializer=bias_init)
    layer = tf.nn.tanh(layer)
    for j in range(depth-1):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="C_"+str(j+1),
                        kernel_initializer=weight_init[j+1],
                        bias_initializer=bias_init)
        layer = tf.nn.tanh(layer)
        layer = tf.layers.dropout(layer, rate=0.05)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="C_last",
                            kernel_initializer=weight_init[depth],
                            bias_initializer=bias_init)
    C_out = layer
    return C_out

def N_net():
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="N_0",
                                kernel_initializer=weight_init[0],
                                bias_initializer=bias_init)
    layer = tf.nn.tanh(layer)
    for j in range(depth-1):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="N_"+str(j+1),
                        kernel_initializer=weight_init[j+1],
                        bias_initializer=bias_init)
        layer = tf.nn.tanh(layer)
        layer = tf.layers.dropout(layer, rate=0.05)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="N_last",
                            kernel_initializer=weight_init[depth],
                            bias_initializer=bias_init)
    N_out = layer
    return N_out

def H_net(): 
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="H_0",
                                kernel_initializer=weight_init[0],
                                bias_initializer=bias_init)
    layer = tf.nn.tanh(layer)
    for j in range(depth-1):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="H_"+str(j+1),
                        kernel_initializer=weight_init[j+1],
                        bias_initializer=bias_init)
        layer = tf.nn.tanh(layer)
        layer = tf.layers.dropout(layer, rate=0.05)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="H_last",
                            kernel_initializer=weight_init[depth],
                            bias_initializer=bias_init)
    H_out = layer
    return H_out

def sapt_net(sym_input,atoms):
    atom_inps = (sym_input, atoms) 
    atom_outs = tf.map_fn(route_atom,atom_inps,dtype="float32")
    output = tf.reduce_sum(atom_outs)
    return output

def route_atom(atom_inps):
    sym_inp = atom_inps[0]
    atomtype = atom_inps[1]
    bool_vec = tf.equal(unique_atoms,atomtype)
    is_O = tf.gather_nd(bool_vec,[0])
    is_C = tf.gather_nd(bool_vec,[1])
    is_N = tf.gather_nd(bool_vec,[2])
    is_H = tf.gather_nd(bool_vec,[3])
    atom_out = tf.contrib.framework.smart_case({is_O:O_net,
                                        is_C:C_net,
                                        is_N:N_net,
                                        is_H:H_net},
                                        exclusive=True,name="route")
    return atom_out


if __name__ == "__main__":
    NN = symfun_parameters.NNforce_field("GA_opt", 0, 0)
    sym_param = NN.element_force_field["H"] #generalize this later
    atom_inp_size = len(sym_param.radial_symmetry_functions) + len(
                    sym_param.angular_symmetry_functions)
    (filenames, sym_input, atoms_in,
        labels, max_atoms, unique_atoms) = input_fn(["./testinp"])
    #atom_nets, unique_atoms = build_atom_nets(unique_atoms,atom_inp_size)
    width = 20
    depth = 2
    training_epochs = 45
    batch_size = 1
    display_step = 1
    
    unique_atoms = tf.cast(unique_atoms, dtype=tf.string)
    symfuns = tf.placeholder(tf.float32, shape=[None, 32])
    atoms = tf.placeholder(tf.string, shape=[None])
    en_label = []
    for i in range(len(labels)):
        en_label.append(labels[i][0])

    weight_init = [tf.random_normal_initializer(stddev=np.sqrt(2.0/32.0))]
    for i in range(depth):
        weight_init.append(tf.random_normal_initializer(stddev=np.sqrt(2.0/width)))
    
    bias_init = tf.random_normal_initializer(0.0,stddev=0.0)
    output = sapt_net(symfuns,atoms)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("summaries", sess.graph)

        label = tf.placeholder(tf.float32, shape=[])
        loss = tf.losses.mean_squared_error(output,label)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0006).minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        t1 = time.time()
        for epoch in range(training_epochs):
            avg_loss = 0.0
            #total_batch = int(len(sym_input) / batch_size)
            #sym_batches = np.array_split(sym_input, total_batch)
            #atom_batches = np.array_split(atoms_in, total_batch)
            #y_batches = np.array_split(en_label, total_batch)
            
            atom_choice = []
            for i in range(len(sym_input)):
                sym = np.array(sym_input[i])
                atom = np.array(atoms_in[i])
                y = en_label[i]

                _, c = sess.run([optimizer, loss], 
                                feed_dict = {symfuns: sym,
                                            atoms: atom,
                                            label: y})
                avg_loss += c / len(sym_input)
            if epoch % display_step == 0:
                t2 = time.time()
                elapsed = (t2-t1)/60
        inputs = {"symfun_ph":symfuns, "atom_ph":atoms,"label_ph":label}
        outputs = {"prediction":output}
        tf.saved_model.simple_save(sess, "./testing_model", inputs, outputs)

