import routines
import time
import numpy as np
import tensorflow as tf
import symmetry_functions as sym
import FFenergy_openMM as saptff
import symfun_parameters
from tensorflow.python import debug as tf_debug

def input_fn(inputdirs):
    for i in range(len(inputdirs)):
        sym_input = []
        path = inputdirs[i]
        NN = symfun_parameters.NNforce_field('GA_opt', 0, 0)
        (atom_tens,atom_num_tens,xyz) = routines.get_xyz_from_combo_files(path)
        (filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)
        max_atoms = tf.constant(get_max_atoms(atom_tens))
        symfun_files = [] 
        labels = []
        atoms = []
        atom_nums = []
        for i in range(len(filenames)):
            file = f"{path}/{filenames[i]}_symfun.npy"
            symfun_files.append(file)
            sym_input.append(np.load(file))
            #sym_input.append(tf.convert_to_tensor(np.load(file)))
            label = [tot_en[i], elst[i], exch[i], ind[i], disp[i]]
            labels.append(label)
            #labels.append(tf.convert_to_tensor(label))
            atoms.append(atom_tens[i])
            #atoms.append(tf.convert_to_tensor(atom_tens[i]))
            atom_nums.append(tf.convert_to_tensor(atom_num_tens[i]))
        #sym_input = routines.scale_symmetry_input(sym_input)
    num_unique, atom_indices, unique_atoms = routines.create_atype_list(
                                        atom_tens,routines.atomic_dictionary())
    #sym_input = tf.cast(tf.stack(sym_input),dtype=tf.float32)
    #dataset = tf.data.Dataset.from_tensor_slices((sym_input,atoms,labels))
    #later do:  dataset = dataset.batch(batch_size)
    #           dataset = dataset.prefetch(1)
    return filenames, sym_input, atoms, labels, max_atoms, unique_atoms

def get_max_atoms(atoms):
    max_atoms = 0
    for i in range(len(atoms)):
        if len(atoms[i]) > max_atoms:
            max_atoms = len(atoms[i])
    return max_atoms

"""
def build_atom_nets(unique_atoms, atom_inp_size):
    atom_nets = []
    depth = 3
    width = 200
    atom_names = tf.cast(unique_atoms,tf.string)


    for i in range(len(unique_atoms)):
        atom_net = []
        atom_inp = tf.placeholder(tf.float32, shape=[None,atom_inp_size],
                                    name=str(i)+"_inp")
        layer = tf.layers.dense(atom_inp, width, activation='relu',
                                use_bias=True, name=str(i)+"_0")#(atom_inp)
        print(layer)
        atom_net.append(layer)
        for j in range(depth-1):
            layer = tf.layers.dense(layer, width, activation='relu',
                                use_bias=True, name=str(i)+"_"+str(j+1))
            atom_net.append(layer)
        #atom_net = tf.stack(atom_net)
        atom_nets.append(atom_net)
    
    unique_atoms = tf.cast(unique_atoms, dtype=tf.string)
    #atom_nets = tf.stack(atom_nets)
    return atom_nets, unique_atoms
"""

def O_net():
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="O_0",
                                kernel_initializer=weight_init)
    layer = tf.nn.leaky_relu(layer)
    for j in range(depth):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="O_"+str(j+1),
                        kernel_initializer=weight_init)
        layer = tf.nn.leaky_relu(layer)
        #layer = tf.layers.dropout(layer, rate=0.1)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="O_last", kernel_initializer=weight_init)
    #layer = tf.layers.dropout(layer, rate=0.1)
    O_out = layer
    return O_out

def C_net():
    
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="C_0",
                                kernel_initializer=weight_init)
    layer = tf.nn.leaky_relu(layer)
    for j in range(depth):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="C_"+str(j+1),
                        kernel_initializer=weight_init)
        layer = tf.nn.leaky_relu(layer)
        #layer = tf.layers.dropout(layer, rate=0.1)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="C_last", kernel_initializer=weight_init)
    #layer = tf.layers.dropout(layer, rate=0.1)
    C_out = layer
    return C_out

def N_net():
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="N_0",
                                kernel_initializer=weight_init)
    layer = tf.nn.leaky_relu(layer)
    for j in range(depth):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="N_"+str(j+1),
                        kernel_initializer=weight_init)
        layer = tf.nn.leaky_relu(layer)
        #layer = tf.layers.dropout(layer, rate=0.1)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="N_last", kernel_initializer=weight_init)
    #layer = tf.layers.dropout(layer, rate=0.1)
    N_out = layer
    return N_out

def H_net(): 
    layer = tf.layers.dense(symfuns, width, activation=None,
                                use_bias=True, reuse=tf.AUTO_REUSE, name="H_0",
                                kernel_initializer=weight_init)
    layer = tf.nn.leaky_relu(layer)
    for j in range(depth):
        layer = tf.layers.dense(layer, width, activation=None,
                        use_bias=True, reuse=tf.AUTO_REUSE,name="H_"+str(j+1),
                        kernel_initializer=weight_init)
        layer = tf.nn.leaky_relu(layer)
        #layer = tf.layers.dropout(layer, rate=0.1)
    layer = tf.layers.dense(layer, 20, use_bias=True, reuse=tf.AUTO_REUSE,
                            name="H_last", kernel_initializer=weight_init)
    #layer = tf.layers.dropout(layer, rate=0.1)
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
        labels, max_atoms, unique_atoms) = input_fn(["NMA-Aniline_Step_3"])
    #atom_nets, unique_atoms = build_atom_nets(unique_atoms,atom_inp_size)
    width = 30
    depth = 2
    unique_atoms = tf.cast(unique_atoms, dtype=tf.string)
    symfuns = tf.placeholder(tf.float32, shape=[None, 32])
    atoms = tf.placeholder(tf.string, shape=[None])
    weight_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/width))
    output = sapt_net(symfuns,atoms)
 
    training_epochs = 500
    batch_size = 32
    display_step = 1
   
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("summaries", sess.graph)
        en_label = []
        for i in range(len(labels)):
            en_label.append(labels[i][0])
        label = tf.placeholder(tf.float32, shape=[])
        cost = tf.losses.mean_squared_error(output,label)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        init = tf.global_variables_initializer()
        sess.run(init)
        t1 = time.time()
        for epoch in range(training_epochs):
            avg_cost = 0.0
            #total_batch = int(len(sym_input) / batch_size)
            #sym_batches = np.array_split(sym_input, total_batch)
            #atom_batches = np.array_split(atoms_in, total_batch)
            #y_batches = np.array_split(en_label, total_batch)
            
            atom_choice = []
            for i in range(len(sym_input)):
                sym = np.array(sym_input[i])
                atom = np.array(atoms_in[i])
                y = en_label[i]
                _, c = sess.run([optimizer, cost], 
                                feed_dict = {symfuns: sym,
                                            atoms: atom,
                                            label: y})
                avg_cost += c / len(sym_input)
            if epoch % display_step == 0:
                t2 = time.time()
                elapsed = (t2-t1)/60
                print("Epoch", '%04d' % (epoch+1), "| cost=", \
                    "{:.9f}".format(avg_cost), f"| time = {elapsed} min")
        inputs = {"symfun_ph":symfuns, "atom_ph":atoms,"label_ph":label}
        outputs = {"prediction":model_output}
        tf.saved_model.simple_save(sess, "./saved_model", inputs, outputs)
