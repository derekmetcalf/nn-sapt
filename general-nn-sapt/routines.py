import os
import sys
import glob
import numpy as np
import time
from numpy import linalg as LA
import symmetry_functions as sym
from joblib import Parallel, delayed
from symfun_parameters import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import losses
import tensorflow as tf
"""
Perform miscellaneous standard functions of the NN-SAPT paradigm.

"""


class DataSet:
    
    """Create data set object for arbitrary SAPT data collections.

    This class is very fidgety and has lots of artifacts 
    to include handling for certain obscure cases. Almost any dataset 
    format that has not been explicitly coded into this class either 
    needs a new custom method or a generalization of an old method.

    """

    def __init__(self, path):
        """Initialize SAPT dataset object.

        How not to write Python: a tutorial. This constructor includes
        suitable tensors for usual SAPT neural network training tasks.
        Additions should be made with great consideration about the
        data source, since there is minimal handling for erroneous
        objects.

        """
        self.path = path
        self.aname = []
        self.atom_tensor = []
        self.xyz = []
        self.elec = []
        self.ind = []
        self.disp = []
        self.exch = []
        self.delHF = []
        self.energy = []
        self.geom_files = []

    def read_QM9_data():
        """Get energy and XYZ information from QM9 dataset.

        Method walks through the directory containing QM9 (standard from 
        GitHub), parses files, and writes to convenient tensors.

        """
        for root, dirs, files in os.walk("./%s" % self.target_dir):
            for target in files:
                geom_file = "./%s/%s" % (self.target_dir, target)
                if os.path.exists(geom_file):
                    file = open(geom_file, "r", errors="ignore")
                    line_num = 0
                    dimeraname = []
                    dimerxyz = []
                    atoms_iterated = 0
                    for line in file:
                        line = line.replace("   ", " ")
                        if line_num == 0:
                            num_atoms = int(line.strip())
                            #print(num_atoms)
                        if line_num == 1:
                            data = line.split()
                            energy.append(float(data[12]))
                        if line_num > 1 and atoms_iterated < num_atoms:
                            data = line.split()
                            dimeraname.append(data[0])
                            for k in [1, 2, 3]:
                                if "*^" in str(data[k]):
                                    data[k] = data[k].replace("*^", "e")
                            dimerxyz.append([
                                float(data[1]),
                                float(data[2]),
                                float(data[3])
                            ])
                            #print(dimerxyz)
                            atoms_iterated += 1
                        line_num += 1
                    #dimerxyz = np.array(dimerxyz)
                    aname.append(dimeraname)
                    xyz.append(dimerxyz)
                    file.close()
        # change lists to arrays
        self.xyz = np.array(xyz, dtype=object)
        self.energy = np.array(energy)
        self.atom_tensor = self.aname
        return self.dimeraname, self.xyz, self.energy, self.atom_tensor, self.en_units


    def read_sapt_from_target_dir(self):
        """Collect energies and geometries from disparate sources.

        This is a unitask method for property scraping in strange 
        situations like files stored with different naming conventions 
        or combining many data directories.

        """
        path = self.path

        #add path handling here to get target_list and the xyz_path

        with open(target_list, "r") as file:
            next(file)
            for line in file:
                molec = line.split(",")
                filename = molec[0]

                geom_file = "%s/%s"%(self.xyz_path,filename.replace(".out",".out.xyz"))
                if os.path.exists(geom_file):
                    geom = open(geom_file, "r")
                    line_num = 0
                    dimeraname = []
                    dimerxyz = []
                    atoms_iterated = 0
                    for line in geom:
                        if line_num == 0:
                            num_atoms = int(line.strip())
                        elif line_num == 1:
                            line_num += 1
                            continue
                        elif line_num > 1 and atoms_iterated < num_atoms:
                            data = line.split()
                            dimeraname.append(data[0])
                            dimerxyz.append([
                                float(data[1]),
                                float(data[2]),
                                float(data[3])
                            ])
                            atoms_iterated += 1
                        line_num += 1
                    self.aname.append(dimeraname)
                    self.xyz.append(dimerxyz)
                    self.geom_files.append(filename)
                    geom.close()

                else:
                    print("Invalid geometry file path encountered")
                    print(geom_file)

                self.energy.append(molec[1])
                self.elec.append(molec[2])
                self.exch.append(molec[3])
                self.ind.append(molec[4])
                self.disp.append(molec[5])
                self.atom_tensor = self.aname


        return (self.aname, self.atom_tensor, self.xyz, self.elec,
                self.ind, self.disp, self.exch, self.energy,
                self.geom_files)
    
    def read_from_xyz_dir(self):
        (self.xyz, self.aname, self.geom_files) = get_xyzs_from_dir(self.target_dir)
        (self.energy, self.elec, self.exch,
            self.ind, self.disp) = get_energies_from_list(self.prop_path,
                                                            self.geom_files)
    
        return (self.aname, self.xyz, self.energy,
                self.elec, self.exch, self.ind, self.disp,
                self.geom_files)

def get_test_mask(atom_nums, train_atom_nums):
    seen_atoms = []
    atom_ind_dict = {}
    num_atomtypes = -1
    for i in range(len(train_atom_nums)):
        for j in range(len(train_atom_nums[i])):
            if train_atom_nums[i][j] not in seen_atoms:
                num_atomtypes += 1
                seen_atoms.append(train_atom_nums[i][j])
                atom_ind_dict.update({train_atom_nums[i][j] : num_atomtypes})
                
    mask = []
    for i in range(len(atom_nums)):
        molec_mask = []
        for j in range(len(atom_nums[i])):
            atom_mask = np.zeros(num_atomtypes + 1)
            atom_mask[atom_ind_dict[atom_nums[i][j]]] = 1
            molec_mask.append(atom_mask)
        mask.append(molec_mask)
    mask = np.array(mask)
    return mask

def get_connectivity_mat(xyz):
    molec_rij = []
    molec_dr = []
    # first loop over atoms
    for i_atom in range(len(xyz)):
        atom_rij = []
        atom_dr = []
        # second loop over atoms
        for j_atom in range(len(xyz)):
            x_diff = xyz[i_atom][0] - xyz[j_atom][0]
            y_diff = xyz[i_atom][1] - xyz[j_atom][1]
            z_diff = xyz[i_atom][2] - xyz[j_atom][2]
            atom_dr.append([x_diff, y_diff, z_diff])
            atom_rij.append(LA.norm([x_diff, y_diff, z_diff]))
        molec_dr.append(atom_dr)
        molec_rij.append(atom_rij)
    molec_dr = np.array(molec_dr)
    molec_rij = np.array(molec_rij)
    
    connectivity = np.zeros(np.shape(molec_rij))
    for i in range(len(connectivity)):
        for j in range(len(connectivity[i])):
            if molec_rij[i][j] < 1.7:
                connectivity[i][j] = 1
    return connectivity 

def get_mask(atom_nums):
    seen_atoms = []
    atom_ind_dict = {}
    num_atomtypes = -1
    for i in range(len(atom_nums)):
        for j in range(len(atom_nums[i])):
            if atom_nums[i][j] not in seen_atoms:
                num_atomtypes += 1
                seen_atoms.append(atom_nums[i][j])
                atom_ind_dict.update({atom_nums[i][j] : num_atomtypes})
                print(atom_nums[i][j])
    mask = []
    for i in range(len(atom_nums)):
        molec_mask = []
        for j in range(len(atom_nums[i])):
            atom_mask = np.zeros(num_atomtypes + 1)
            atom_mask[atom_ind_dict[atom_nums[i][j]]] = 1
            molec_mask.append(atom_mask)
        mask.append(molec_mask)
    mask = np.array(mask)
    return mask

def pad_sym_inp(sym_inp, max_train_atoms=0):

    max_atoms = 0
    max_symfuns = 0
    for i in range(len(sym_inp)):
        if len(sym_inp[i]) > max_atoms:
            max_atoms = len(sym_inp[i])
        for j in range(len(sym_inp[i])):
            if len(sym_inp[i][j]) > max_symfuns:
                max_symfuns = len(sym_inp[i][j])
    N = len(sym_inp)
    if max_train_atoms == 0:
        padded_inp = np.zeros((N,max_atoms,max_symfuns))
    else: padded_inp = np.zeros((N,max_train_atoms,max_symfuns))
    for i in range(len(sym_inp)):
        for j in range(len(sym_inp[i])):
            for k in range(len(sym_inp[i][j])):
                padded_inp[i][j][k] = sym_inp[i][j][k]
    return padded_inp

def combine_energies_xyz(energy_file, xyz_path, new_path):
    """Given an energy_file  containing a list of files and corresponding
    energies and xyz_path, a path of .xyz files with names corresponding 
    to names in the energy_file, write a new set of .xyz files containing
    the energies (or other prudent info)  in the comment line to new_path.

    """
    with open(energy_file, "r") as en_file:
        filenames = []
        tot_int = []
        elst = []
        exch = []
        ind = []
        disp = []
        #next(en_file)
        for line in en_file:
            line = line.split(",")
            filenames.append(line[0].replace(".out",".out.xyz"))
            tot_int.append(line[1])
            elst.append(line[2])
            exch.append(line[3])
            ind.append(line[4])
            disp.append(line[5])
        en_file.close()
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    for i in range(len(filenames)):
        data = []
        with open(f"{xyz_path}/{filenames[i]}","r") as old_xyz:
            for line in old_xyz:
                data.append(line)
        with open(f"{new_path}/{filenames[i]}","w+") as new_xyz:
            new_xyz.write(data[0])
            new_xyz.write(f"{filenames[i]},{tot_int[i]},{elst[i]},{exch[i]},{ind[i]},{disp[i]}")
            for j in range(len(data)-2):
                new_xyz.write(data[j+2])
    return

def get_xyz_from_combo_files(path, filenames):
        
    atom_dict = atomic_dictionary()
    xyz = []
    atoms = []
    atom_nums = []
    #for file in glob.glob(f"{path}/*.xyz"):
    for filename in filenames:
        if ".xyz" in filename:
            file = f"{path}/{filename}"
        else:
            file = f"{path}/{filename}.xyz"
        with open(file, "r") as xyzfile:
            next(xyzfile)
            next(xyzfile)
            molec_atoms = []
            molec_xyz = []
            molec_atom_nums = []
            #print(file)
            for line in xyzfile:
                data = line.split()
                molec_atoms.append(data[0])
                #print(data[0])
                molec_atom_nums.append(atom_dict[data[0]])
                molec_xyz.append([float(data[1]), float(data[2]), float(data[3])])
            atoms.append(molec_atoms)
            xyz.append(molec_xyz)
            atom_nums.append(molec_atom_nums)
    return atoms, atom_nums, xyz

def get_sapt_from_combo_files(path, keep_prob=1.00):
    """Collect SAPT energies and corresponding filenames from the combo xyz
    file / energy files written by routines.combine_energies_xyz

    """
    filenames = []
    tot_en = []
    elst = []
    exch = []
    ind = []
    disp = []
    split_vec = []
    for file in glob.glob(f"{path}/*.xyz"):
        rand_num = np.random.rand()
        if rand_num > keep_prob:
            continue
        #print(file)
        with open(file, "r") as saptfile:
            temp = saptfile.readlines()
            data = temp[1].split(",")
            filenames.append(data[0])
            tot_en.append(float(data[1]))
            elst.append(float(data[2]))
            exch.append(float(data[3]))
            ind.append(float(data[4]))
            disp.append(float(data[5]))
            if len(data) == 7:
                split_vec.append(int(data[6].replace("\n","")))
            else: split_vec.append(0)
    return filenames, tot_en, elst, exch, ind, disp, split_vec

def make_perturbed_xyzs(path, max_shift=0.02):
    """Currently, make a perturbed xyz file for each file in the target
    directory and deposits those files in the same path.

    """
    (atoms,xyz) = get_xyz_from_combo_files(path)
    (filenames,tot_int,elst,exch,ind,disp) = get_sapt_from_combo_files(path)
    for i in range(len(xyz)):
        perturbation = (np.random.rand(len(xyz[i]),3) - 0.5) * 2 * max_shift
        new_xyz = xyz[i] + perturbation
        new_filename = "supp_" + filenames[i]
        with open(f"{path}/{new_filename}","w+") as xyz_file:
            xyz_file.write(f"{len(new_xyz)}\n")
            xyz_file.write(f"{new_filename},{tot_int[i]},{elst[i]},{exch[i]},{ind[i]},{disp[i]}\n")
            for j in range(len(new_xyz)):
                xyz_file.write(f"{atoms[i][j]} {new_xyz[j,0]} {new_xyz[j,1]} {new_xyz[j,2]}\n")
                
        
    return

def get_energies_from_list(file_path, target_files):
    """Scrape energies given a list of target files and a path to 
    a file listing their energy.

    """
    files_searched = 0
    tot = []
    elst = []
    exch = []
    ind = []
    disp = []
    with open(file_path, "r") as infile:
        for target_file in target_files:
            files_searched += 1
            match_name = target_file + ".trunc.out"
            for line in infile:
                data = line.split(",")
                if match_name in data:
                    tot.append(data[1])
                    elst.append(data[2])
                    exch.append(data[3])
                    ind.append(data[4])
                    disp.append(data[5])
            if len(tot) != files_searched:
                print("Couldn't find file")
                quit()
    return tot,elst,exch,ind,disp            

def scan_directory(target_directory, target_file):
    """Scan a directory for target file"""
    for path, dirs, files in os.walk("./%s" % target_directory):
        for f in files:
            if f == target_file and f is not None:
                file_loc = os.path.join(path, f)
                #print(f)
    if "file_loc" in locals():
        return file_loc
    if not "file_loc" in locals():
        print("File location not found in directory")


def get_xyzs_from_dir(directory):
    """Given a directory with xyz files, scrape and return all data in lists"""
    xyz = []
    file_list = []
    aname = []
    for filename in os.listdir(directory):
        file_list.append(filename.replace(".xyz",""))
        with open("%s/%s"%(directory,filename), "r") as infile:
            linenum=0
            molec_xyz = []
            molec_aname = []
            for line in infile:
                if linenum>1 and len(line.split()) == 4:
                    data = line.split()
                    molec_aname.append(data[0])
                    molec_xyz.append([data[1],data[2],data[3]])
                linenum += 1
        infile.close()
        aname.append(molec_aname)
        xyz.append(molec_xyz)
    print(len(xyz))
    print("Got xyzs & file names")
    return xyz, aname, file_list

def progress(count, total, suffix=''):
    """Print a rolling progress bar to standard out."""
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def infer_on_test(model, sym_inp):
    """Perform inferences symmetry function input given a Keras model."""
    #print(sym_inp)
    (en_pred, elec_pred, exch_pred, ind_pred,
        disp_pred) = model.predict_on_batch(sym_inp)
    return en_pred, elec_pred, exch_pred, ind_pred, disp_pred


def error_statistics(predicted, actual):
    """Get MAE, RMSE, and maximum error for arbitrary scalar error vectors."""
    error = np.zeros(len(actual))
    error_sq = np.zeros(len(actual))
    predicted = np.asarray(predicted).astype(float)
    actual = actual.astype(float)
    for i in range(len(actual)):
        error[i] = np.absolute(actual[i] - predicted[i])
        error_sq[i] = (actual[i] - predicted[i])**2
    mae = np.sum(error) / len(actual)
    rmse = np.sqrt(np.sum(error_sq) / len(actual))
    max_error = np.amax(error)
    return mae, rmse, max_error


def get_atomic_num_tensor(atom_tensor, atom_dict):
    """Build list of lists of atomic numbers for a dataset."""
    atomic_num_tensor = []
    for i in range(len(atom_tensor)):
        molec_atom_num_list = []
        for j in range(len(atom_tensor[i])):
            molec_atom_num_list.append(atom_dict[atom_tensor[i][j]])
        atomic_num_tensor.append(molec_atom_num_list)
    return atomic_num_tensor


def set_modeller_coordinates(pdb, modeller, xyz_in):
    """Set the coordinates of the OpenMM 'pdb' and 'modeller' objects.
    
    Takes input coordinate array 'xyz_in'.

    """
    # loop over atoms
    for i in range(len(xyz_in)):
        # update pdb positions
        pdb.positions[i] = Vec3(xyz_in[i][0], xyz_in[i][1],
                                xyz_in[i][2]) * nanometer

    # now update positions in modeller object
    modeller_out = Modeller(pdb.topology, pdb.positions)

    return modeller_out


def slice_arrays(arrays, start=None, stop=None):
    """Slice array with specified bounds."""
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def create_atype_list(aname, atomic_dict):
    """Create a lookup list of unique element indices.

    'atype' is a lookup list of numeric indices.
    'aname' is the tensor of strings of atomic identifiers like 'C' or 'H'.
    
    """
    index = 0
    atype = []
    unique_list = []
    for i in range(len(aname)):
        atype_molecule = []
        for j in range(len(aname[i])):
            if aname[i][j] not in unique_list:
                unique_list.append(aname[i][j])
            atype_molecule.append(unique_list.index(aname[i][j]))
        atype.append(atype_molecule)
    #print(unique_list)
    ntype = len(unique_list)
    return ntype, atype, unique_list

def get_permutation_vector(test_atom_list, train_atom_list):
    """Permute / add to atomtype vecs to rectify test set symfun ordering"""
    permute_vec = []
    not_in_set = []
    for i, atomtype in enumerate(test_atom_list):
        if atomtype in train_atom_list:
            train_atom_index = train_atom_list.index(atomtype)
            permute_vec.append(train_atom_index)
        else:
            print("Error: Trying to test on atoms outside the training set")
            quit()
    while len(train_atom_list) > len(permute_vec):
        permute_vec.append(len(permute_vec))
    return permute_vec

def permute_atoms(sym_input_in, permute_vec, test_unique_atoms):
    diff = len(permute_vec) - len(test_unique_atoms)
    sym_input = np.zeros((sym_input_in.shape[0],sym_input_in.shape[1],sym_input_in.shape[2]))#+diff))
    for l in range(len(permute_vec)):
        permute_vec[l] = permute_vec[l] - len(permute_vec)
    for i in range(len(sym_input_in)):
        for j in range(len(sym_input_in[i])):
            for k in range(len(sym_input_in[i][j])):
                sym_input[i][j][k] = sym_input_in[i][j][k]
            temp_encoding = np.zeros(len(permute_vec))
            for index, permute_index in enumerate(permute_vec):
                temp_encoding[index] = sym_input[i][j][permute_index]
            for index, new_val in enumerate(temp_encoding):
                native_index = index - len(temp_encoding) #- 1
                sym_input[i][j][native_index] = temp_encoding[index]
    return sym_input

def compute_displacements(xyz, path, filenames):
    """Compute displacement tensor from input cartesian coordinates."""
    Parallel(
        n_jobs=4, verbose=1)(delayed(displacement_matrix)(xyz[i], path, i, filenames[i])
                             for i in range(len(xyz)))
    #for i in range(len(xyz)):
    #    displacement_matrix(xyz[i],path,i)
    return


def displacement_matrix(xyz, path, i, filename):
    """Compute displacement matrix for all atoms in a molecule.

    This matrix is size len(xyz) * len(xyz) and saves Euclidian 
    distance (rij) and distance-per-dimenson (dr) tensors to file

    """
    molec_rij = []
    molec_dr = []
    rij_out = "%s/%s_rij.npy" % (path, filename)
    dr_out = "%s/%s_dr.npy" % (path, filename)
    # first loop over atoms
    for i_atom in range(len(xyz)):
        atom_rij = []
        atom_dr = []
        # second loop over atoms
        for j_atom in range(len(xyz)):
            x_diff = xyz[i_atom][0] - xyz[j_atom][0]
            y_diff = xyz[i_atom][1] - xyz[j_atom][1]
            z_diff = xyz[i_atom][2] - xyz[j_atom][2]
            atom_dr.append([x_diff, y_diff, z_diff])
            atom_rij.append(LA.norm([x_diff, y_diff, z_diff]))
        molec_dr.append(atom_dr)
        molec_rij.append(atom_rij)
    np.save(rij_out, np.array(molec_rij))
    np.save(dr_out, np.array(molec_dr))


def compute_thetas(num_systems, path, filenames):
    """Compute theta tensor from displacement tensor in parallel."""
    Parallel(
        n_jobs=4, verbose=1)(
            delayed(theta_tensor_single)(path, i, filenames[i]) for i in range(num_systems))
    return


def theta_tensor_single(path, i, filename):
    """Compute individual theta tensor and save to file.

    This function is called by compute_thetas for a single-molecule angle
    tensor computation, which is then saved to a file typically with the
    same name convention as the distance matrices.

    """
    molec_theta = []
    outfile = "%s/%s_theta.npy" % (path, filename)
    rij = np.load("%s/%s_rij.npy" % (path, filename))
    dr = np.load("%s/%s_dr.npy" % (path, filename))
    for i_atom in range(len(rij)):
        atom_theta = []
        for j_atom in range(len(rij)):
            atom_atom_theta = []
            for k_atom in range(len(rij)):
                r1 = rij[i_atom][j_atom]
                r2 = rij[i_atom][k_atom]
                dr1 = dr[i_atom][j_atom]
                dr2 = dr[i_atom][k_atom]
                if i_atom != j_atom and i_atom != k_atom and j_atom != k_atom:
                    ph_angle = np.dot(dr1, dr2) / r1 / r2
                    if ph_angle > 1: ph_angle = 1
                    if ph_angle < -1: ph_angle = -1
                    atom_atom_theta.append(np.arccos(ph_angle))
                else:
                    atom_atom_theta.append(float(0.0))

            atom_theta.append(atom_atom_theta)
        molec_theta.append(atom_theta)
    np.save(outfile, np.array(molec_theta))
    return

def construct_symmetry_input(NN, path, filenames, aname, ffenergy, 
                                atomic_num_tensor, val_split, 
                                NN_B=None, split_vec=None):
    """Construct symmetry function input for given dataset.
 
    NOTE: We assume we are using a shared layer NN structure, where
    layers are shared for common atomtypes (elements).  We then will concatenate
    networks into single layers. For this structure, we need input data to 
    be a list of arrays, where each array is the input data (data points, 
    symmetry functions) for each atom.

    Thus, the 'sym_input' output of this subroutine is a list of 2D lists 
    of form [data_atom1, data_atom2, data_atom3, etc.] and data_atomi is 
    an array [ j_data_point , k_symmetry_function ]

    "path" specifies dataset path, "ffenergy" is an artifact for delta learning
    over a force field, and the "NN" simply describes the variety of symfun
    we are interested in computing.

    You may pass a split_vec, indicating where to split each molecule into
    monomers so a "split symmetry function" input may be constructed.
    """
    sym_path = path.replace("spatial_info", "sym_inp")
    if os.path.exists(sym_path) == False: 
        #os.mkdir(sym_path)
        pass
    if None in split_vec:
        print("Using standard symmetry functions")
        Parallel(
            n_jobs=4, verbose=1)(delayed(sym_inp_single)(
                sym_path, path, filenames[j_molec], j_molec, aname, NN, atomic_num_tensor[j_molec]) for j_molec in range(len(aname)))
    elif len(split_vec) == len(filenames):
        print("Using intermolecular symmetry functions")
        Parallel(
            n_jobs=4, verbose=1)(delayed(split_sym_inp_single)(
                sym_path, path, filenames[j_molec], j_molec, aname, NN, NN_B, atomic_num_tensor[j_molec], split_vec[j_molec]) for j_molec in range(len(aname)))
        
    else: print("split_vec and filename list length mismatch")
    return


def sym_inp_single(sym_path, path, filename, j_molec, aname, NN, atomic_num_slice):
    """Compute symmetry function nested lists for a single molecule."""
    outfile = "%s/%s_symfun.npy" % (sym_path, filename)
    rij = np.load("%s/%s_rij.npy" % (path, filename))
    theta = np.load("%s/%s_theta.npy" % (path, filename))
    input_atom = []

    for i_atom in range(len(
            aname[j_molec])):  #loop over all atoms in each molec
        typeNN = NN.element_force_field[aname[j_molec][i_atom]]
        input_i = []
        for i_sym in range(len(typeNN.radial_symmetry_functions)):
            #looping over symfuns for this atom
            G1 = sym.radial_gaussian(
                rij, i_atom, typeNN.radial_symmetry_functions[i_sym][0],
                typeNN.radial_symmetry_functions[i_sym][1], typeNN.Rc,
                atomic_num_slice)
            input_i.append(G1)

        for i_sym in range(len(typeNN.angular_symmetry_functions)):
            G2 = sym.angular_gaussian(
                rij, theta, i_atom,
                typeNN.angular_symmetry_functions[i_sym][0],
                typeNN.angular_symmetry_functions[i_sym][1],
                typeNN.angular_symmetry_functions[i_sym][2], typeNN.Rc,
                atomic_num_slice)
            input_i.append(G2)
        input_atom.append(input_i)
    np.save(outfile, np.array(
        input_atom))  #only supports when all atoms have same # of features
    return

def split_sym_inp_single(sym_path, path, filename, j_molec, aname, NN_A, NN_B, atomic_num_slice,split_val=None):
    """Compute split symmetry function nested lists for a single molecule."""
    outfile = "%s/%s_symfun.npy" % (sym_path, filename)
    rij = np.load("%s/%s_rij.npy" % (path, filename))
    theta = np.load("%s/%s_theta.npy" % (path, filename))
    input_atom = []
    for i_atom in range(len(aname[j_molec])):
        typeNN = NN_A.element_force_field[aname[j_molec][i_atom]]
        input_i = []
        for i_sym in range(len(typeNN.radial_symmetry_functions)):
            G1 = sym.split_radial_gaussian(
                rij,i_atom, typeNN.radial_symmetry_functions[i_sym][0],
                typeNN.radial_symmetry_functions[i_sym][1], typeNN.Rc,
                atomic_num_slice, split_val=split_val, cross_terms=False)
            input_i.append(G1)
        for i_sym in range(len(typeNN.angular_symmetry_functions)):
            G2 = sym.split_angular_gaussian(
                rij, theta, i_atom,
                typeNN.angular_symmetry_functions[i_sym][0],
                typeNN.angular_symmetry_functions[i_sym][1],
                typeNN.angular_symmetry_functions[i_sym][2], typeNN.Rc,
                atomic_num_slice, split_val=split_val, cross_terms=False)
            input_i.append(G2)

        typeNN = NN_B.element_force_field[aname[j_molec][i_atom]] 
        for i_sym in range(len(typeNN.radial_symmetry_functions)):
            #looping over symfuns for this atom
            G1 = sym.split_radial_gaussian(
                rij, i_atom, typeNN.radial_symmetry_functions[i_sym][0],
                typeNN.radial_symmetry_functions[i_sym][1], typeNN.Rc,
                atomic_num_slice, split_val=split_val, cross_terms=True)
            input_i.append(G1)

        for i_sym in range(len(typeNN.angular_symmetry_functions)):
            G2 = sym.split_angular_gaussian(
                rij, theta, i_atom,
                typeNN.angular_symmetry_functions[i_sym][0],
                typeNN.angular_symmetry_functions[i_sym][1],
                typeNN.angular_symmetry_functions[i_sym][2], typeNN.Rc,
                atomic_num_slice, split_val=split_val, cross_terms=True)
            input_i.append(G2)
        input_atom.append(input_i)
    np.save(outfile, np.array(
        input_atom))  #only supports when all atoms have same # of features
    return



def construct_flat_symmetry_input(NN, rij, aname, ffenergy):
    """Construct flattened version of symmetry functions.

    This flat version can pretty much only be used when the entire 
    data set has the same # of atoms and atomtypes...
    """
    # output tensor with symmetry function input
    flat_sym_input = []

    # loop over data points
    for i in range(rij.shape[0]):
        # loop over atoms
        input_i = [ffenergy[i]]
        for i_atom in range(len(aname)):
            # get elementNN object for this atomtype
            typeNN = NN.element_force_field[aname[i_atom]]

            # now loop over symmetry functions for this element
            for i_sym in range(len(typeNN.symmetry_functions)):
                # construct radial gaussian symmetry function
                G1 = sym.radial_gaussian(
                    rij[i], i_atom, typeNN.symmetry_functions[i_sym][0],
                    typeNN.symmetry_functions[i_sym][1], typeNN.Rc)
                input_i.append(G1)
        # convert 2D list to array
        input_i = np.array(input_i)
        # append to list
        flat_sym_input.append(input_i)
    flat_sym_input = np.array(flat_sym_input)
    return flat_sym_input


class SymLayer(Layer):
    """/||Under Construction||\ 

    This is to be used as the first layer of a NN that takes rij and 
    associated atomtypes as input in order to construct the symmetry
    functions within a TensorFlow graph for speed and interoperability.
    This will be especially important to do inferences with production-level
    code so that symmetry function construction is not a bottleneck in
    high-throughput screening or dynamics.

    """

    def __init__(self, **kwargs):
        super(SymLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = 1  #again, temporary to avoid passing in a bunch of junk for now
        return shape

    def build(self, input_shape):
        self.width_init = K.ones(1)
        self.shift_init = K.ones(1)
        self.width = K.variable(self.width_init, name="width", dtype='float32')
        self.shift = K.variable(self.shift_init, name="shift", dtype='float32')
        self.trainable_weights = [self.width, self.shift]
        super(SymLayer, self).build(input_shape)

    def call(self, x):
        Rc = 12.0
        PI = 3.14159265359
        # loop over num. radial sym functions requested
        #for i in range(5):
        # loop over atoms in molec
        G = K.zeros(1, dtype='float32')
        self.width = tf.cast(self.width, tf.float32)
        for j_atom in range(
                6):  #this is super specific for water dimer, fixable
            print(x)
            fc = tf.cond(
                x <= Rc, lambda: 0.5 * (K.cos(PI * x) + 1), lambda: 0.0)
            G = G + fc * K.exp(-self.width * (K.pow(
                (x[j_atom] - self.width), 2)))
            #K.concatenate([G],axis=0)
        print(G)
        return G


def custom_loss(y_true, y_pred):
    """Compute multitarget loss.

    This particular loss function is not used since Keras handles multitarget
    losses naturally in the "fit" method. If native TensorFlow becomes
    mandatory, use this as the loss function.
    
    """
    total_en_loss = K.mean(K.square(y_true[0] - y_pred[0]))
    elst_loss = K.mean(K.square(y_true[1] - y_pred[1]))
    exch_loss = K.mean(K.square(y_true[2] - y_pred[2]))
    ind_loss = K.mean(K.square(y_true[3] - y_pred[3]))
    disp_loss = K.mean(K.square(y_true[4] - y_pred[4]))
    comp_weight = 0
    loss = (1 - comp_weight) * total_en_loss + comp_weight * (
        elst_loss + exch_loss + ind_loss + disp_loss)
    loss = total_en_loss
    return loss


def scale_symmetry_input(sym_input):
    """Scale symmetry input to mean 0 std dev 1."""

    num_atoms = 0
    for molec in range(len(sym_input)):
        for atom in range(len(sym_input[molec])):
            num_atoms += 1
    
    scale_params = np.zeros((len(sym_input[0][0]), num_atoms))
    atom_ind = 0
    for molec in range(len(sym_input)):
        for atom in range(len(sym_input[molec])):
            atom_ind +=1
            for sym in range(len(sym_input[molec][atom])):
                scale_params[sym][atom_ind-1] = sym_input[molec][atom][sym]

    means = []
    stds = []
    for i in range(len(scale_params)):
        means.append(np.average(scale_params[i]))
        stds.append(np.std(scale_params[i]))
        scale_params[i] = scale_params[i] - means[i]
        scale_params[i] = scale_params[i]/stds[i]
    
    for molec in range(len(sym_input)):        
        for atom in range(len(sym_input[molec])):
            atom_ind +=1
            for sym in range(len(sym_input[molec][atom])):
                sym_input[molec][atom][sym] = (sym_input[molec][atom][sym] - means[sym]) / stds[sym] 
    return sym_input, means, stds 


def test_symmetry_input(sym_input):
    """Generate histograms of symmetry input for feature selection.

    For a symmetry function to contribute to a neural network
    it must exhibit a non-negligable variance across the data set.
    Here we test the variance of individual symmetry functions across 
    the data set.

    """
    # loop over atoms
    for i_atom in range(len(sym_input)):
        print("symmetry functions for atom ", i_atom)

        # histogram values of every symmetry function
        for i_sym in range(sym_input[i_atom].shape[1]):

            hist = np.histogram(sym_input[i_atom][:, i_sym])

            for i in range(len(hist[0])):
                print(hist[1][i], hist[0][i])
            print("")

    sys.exit()


def atomic_dictionary():
    """Construct dictionary associating atoms with their atomic numbers.

    Lol there has to be a package that does this but I guess I'm good until
    we need Cesium.

    """
    atom_dict = {
        'H': 1,
        'He': 2,
        'Li': 3,
        'Be': 4,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Ne': 10,
        'Na': 11,
        'Mg': 12,
        'Al': 13,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17,
        'Ar': 18,
        'K': 19,
        'Ca': 20,
        'Sc': 21,
        'Ti': 22,
        'V': 23,
        'Cr': 24,
        'Mn': 25,
        'Fe': 26,
        'Co': 27,
        'Ni': 28,
        'Cu': 29,
        'Zn': 30,
        'Ga': 31,
        'Ge': 32,
        'As': 33,
        'Se': 34,
        'Br': 35,
        'Kr': 36,
        'Rb': 37,
        'Sr': 38,
        'Y': 39,
        'Zr': 40,
        'Nb': 41,
        'Mo': 42,
        'Tc': 43,
        'Ru': 44,
        'Rh': 45,
        'Pd': 46,
        'Ag': 47,
        'Cd': 48,
        'In': 49,
        'Sn': 50,
        'Sb': 51,
        'Te': 52,
        'I': 53,
        'Xe': 54
    }
    return atom_dict
