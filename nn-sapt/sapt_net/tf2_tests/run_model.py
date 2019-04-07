import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sapt_net
import routines


export_dir = "./model_20node_2hlayer_leaky"

filenames, symfuns, atoms, ens, max_atoms, unique_atoms = sapt_net.input_fn(["NMA-Aniline-crystallographic_sym_inp"])

label = []
for i in range(len(filenames)):
    label.append(ens[i][0])

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"],export_dir)
    graph = tf.get_default_graph()
    #print(graph.get_operations())
    #summary = tf.summary.FileWriter(graph)
    #summary.add_graph(graph)
    out = []
    for i in range(len(filenames)):
        out.append(sess.run("Sum:0",feed_dict={"Placeholder:0":symfuns[i], "Placeholder_1:0":atoms[i],"Placeholder_2:0":label[i]}))
    print(np.array(out))
    print(np.array(label)) 
    error_mat = np.array(out)-np.array(label)
    mae = np.average(np.abs(error_mat))
    print(mae)
    plt.scatter(np.array(out),np.array(label))
    plt.show()
