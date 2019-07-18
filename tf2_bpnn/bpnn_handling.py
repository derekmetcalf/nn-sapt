import numpy as np
import tensorflow as tf

class BPNN:
    def __init__(self, optimizer, batch_size, epochs, nodes, train_atoms, 
                coeffs, sym_params, compute_loss, activation='relu', models=[]):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.nodes = nodes
        self.train_atoms = train_atoms
        self.coeffs = coeffs
        self.sym_params = sym_params
        self.models = models
        self.activation = activation
        self.compute_loss = compute_loss

    def build_atom_net(self):
        """ Create a neural network for a single atom type """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(26,), dtype=tf.float64))
        for i, num_nodes in enumerate(self.nodes):
            #if i == 1:
            #    model.add(tf.keras.layers.Dropout(0.02))
            model.add(tf.keras.layers.Dense(self.nodes[i], self.activation))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
  
        model.build()
        return model
    
    def build_atom_nets(self):
        """ Loop over training atoms to construct individual atom nets """
        self.models = [self.build_atom_net() for i in range(len(self.train_atoms))]
    
    def prep_sym_params(self):
        """ Tensorify symfun params for use as graph input """
        self.sym_params = tf.convert_to_tensor(self.sym_params, dtype=tf.float64)

    def load_models(self, model_loc, train_atoms):
        """ When using class for testing, load premade models from path """
        for atom in train_atoms:
            atom_path = model_loc + "/" + atom# + ".h5"
            model = tf.saved_model.load(atom_path)
            self.models.append(model)
        print(f"Models loaded from {model_loc}")

def load_params(path):
    """ Search path for info.txt and extract symfun param matrix """
    with open(path + "/info.txt","r") as info:
        lines = info.readlines()
        gather = False
        params = []
        for line in lines:
            if gather == True:
                line_params = line.replace("[","")
                line_params = line_params.rstrip()
                line_params = line_params.replace("]","")
                line_params = np.array(line_params.split(), dtype=np.float)
                params.append(line_params)
                if "]]" in line: gather = False           
            if "Symfun parameters" in line:
                gather = True
        info.close()
    params = np.array(params)
    return params

def load_coeffs(path):
    """ Search path for info.txt and extract train atoms and coeffs """
    with open(path + "/info.txt","r") as info:
        lines = info.readlines()
        gather = False
        atoms = []
        coeffs = []
        for line in lines:
            if "Network nodes" in line: gather = False
            if gather == True:
                parsed = line.strip().split()
                atom, coeff = parsed[0], float(parsed[1])
                atoms.append(atom)
                coeffs.append(coeff)
            if "Atomic linear regression parameters" in line:
                gather = True
        info.close()
    return atoms, coeffs

def train_test_check(train_Z, test_Z):
    for i, Z in enumerate(train_Z):
        if Z in test_Z:
            if Z == test_Z[i]:
                pass
            else:
                raise ValueError("Train and test atom indeces misaligned.")
        else:
            print("Atom number {Z} in train set but not test set.")
    for i, Z in enumerate(test_Z):
        if Z not in train_Z:
            raise ValueError(f"Atom number {Z} in test set but not train set.")
    return
