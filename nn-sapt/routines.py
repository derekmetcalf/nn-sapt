#from simtk.openmm.app import *
#from simtk.openmm import *
#from simtk.unit import *
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
#import sapt_SSI
#import pandas as pd
"""
these are general subroutines that we use for the NN program
these subroutines are meant to be indepedent of tensorflow/keras libraries,
and are used for I/O, data manipulation (construction of symmetry functions), etc.
"""



class DataSet:
    def __init__(self, target_dir, mode, en_units):
        self.target_dir = target_dir
        self.mode = mode
        self.en_units = en_units
        if self.mode != "alex":
            self.aname = []
            self.atom_tensor = []
            self.xyz= []
            self.elec = []
            self.ind = []
            self.disp = []
            self.exch = []
            self.delHF = []
            self.energy = []
            self.geom_files = []
        elif self.mode == "alex":    
            self.training_aname = []
            self.training_atom_tensor = []
            self.training_xyz= []
            self.training_elec = []
            self.training_ind = []
            self.training_disp = []
            self.training_exch = []
            self.training_delHF = []
            self.training_energy = []
            self.training_geom_files = []
            self.testing_aname = []
            self.testing_atom_tensor = []
            self.testing_xyz= []
            self.testing_elec = []
            self.testing_ind = []
            self.testing_disp = []
            self.testing_exch = []
            self.testing_delHF = []
            self.testing_energy = []
            self.testing_geom_files = []

    def read_SSI(inputfile):
        """
        reads SSI geometries with energy labels
        """
        self.ssi_data = pd.DataFrame(sapt_SSI.sapt_data())
        
        f = open(inputfile, "r")
        geom_mode = 0
        for line in f:
            if "GEOS[" in line:
                self.dimername = line.split("'")[3]
                self.temp_xyz = []
                geom_line = 0
            if line[:1]=="C " or line[:1]=="H " or line[:1]=="O " or line[:1]=="N " or line[:1]=="S ":
                self.this_line = line.split(" ")
        return

    def read_pdb_sapt_data(inputfile):
        """
        reads SAPT interaction energy input file, outputs
        list of all atomic coordinates, and total interaction energies
        """
    
        file = open(inputfile)
        for line in file:
            # atoms first molecule
            atoms=int(line.rstrip())
            
            # empty list for this dimer
            dimeraname=[]
            dimerxyz=[]
            dimerxyzD=[]
            # first molecule
            for i in range(atoms):
                line=file.readline()
                data=line.split()
                # add xyz to dimer list
                dimeraname.append( data[0] )
                dimerxyz.append([float(data[1]), float(data[2]), float(data[3])])
            # pop dummy atom off list
            dimeraname.pop()
            dimerxyz.pop()
            # second molecule
            line=file.readline()
            atoms=int(line.rstrip())
            for i in range(atoms):
                line=file.readline()
                data=line.split()
                # add xyz to dimer list
                dimeraname.append(data[0])
                dimerxyz.append([float(data[1]), float(data[2]), float(data[3])])
            # pop dummy atom off list
            dimeraname.pop()
            dimerxyz.pop()
            aname.append(dimeraname)
            xyz.append( dimerxyz )
            # now read energy SAPT terms
            for i in range(19):
                line=file.readline()
                data = line.split()
                # get piecewise energies
                if i == 0:
                    val = float(data[1])
                    E1pol = val
                elif i == 1:
                    val = float(data[1])
                    E1exch = val
                elif i == 4:
                    val = float(data[1])
                    E2ind = val
                elif i == 5:
                    val = float(data[1])
                    E2ind_exch = val
                elif i == 7:
                    val = float(data[1])
                    E2disp = val
                elif i == 9:
                    val = float(data[1])
                    E2disp_exch = val
                # get delta HF energy
                elif i == 17:
                    val = float(data[1])
                    dhf= val
            E1tot = E1pol + E1exch
            E2tot = E2ind + E2ind_exch + E2disp + E2disp_exch
            etot = E1tot + E2tot + dhf
            E_elec = E1pol
            E_exch = E1exch
            E_ind = E2ind + E2ind_exch
            E_disp = E2disp + E2disp_exch
            elec.append(E_elec)
            exch.append(E_exch)
            ind.append(E_ind)
            disp.append(E_disp)
            delHF.append(dhf)
            energy.append(etot)
        # change lists to arrays
        xyz=np.array(xyz)
        elec = np.array(elec)
        exch = np.array(exch)
        ind = np.array(ind)
        disp = np.array(disp)
        delHF = np.array(delHF)
        energy=np.array(energy)
        dimername = aname[0]
        atom_tensor = np.asarray(aname)
        return dimeraname, xyz, elec, exch, ind, disp, delHF, energy, atom_tensor
    
    def scan_directory(target_directory, target_file):
        for path, dirs, files in os.walk("./%s"%target_directory):
            for f in files:
                if f == target_file and f is not None:
                    file_loc = os.path.join(path, f)
                    print(f)
        if "file_loc" in locals():
            return file_loc
        if not "file_loc" in locals():
            print("File location not found in directory")
    
    def read_sapt_data_from_csv(csv, target_directory):
        """
        Scrapes all SAPT output files in target_directory 's subdirectories
        Reports molecular geometries and their SAPT energy decomposition labels
        """ 
        with open(csv) as f1:
            count = 1
            found = 0
            for line in f1:
                if count > 1:
                    data = line.split(",")
                    if grab_xyz_from_psi4out(target_directory,data[0]) is not None:
                        dimeraname,dimerxyz = grab_xyz_from_psi4out(target_directory,data[0])
                        found +=1
                        outfile_name.append(data[0])
                        aname.append(dimeraname)
                        xyz.append(dimerxyz)
                        energy.append(data[1])
                        elec.append(data[2])
                        exch.append(data[3])
                        ind.append(data[4])
                        disp.append(data[5])
                count += 1
                print(count)
            f1.close()
    
        #for outfile in outfile_name:
    
        # change lists to arrays
        self.xyz=np.array(self.xyz)
        self.elec = np.array(self.elec)
        self.exch = np.array(self.exch)
        self.ind = np.array(self.ind)
        self.disp = np.array(self.disp)
        self.energy=np.array(self.energy)
        self.dimername = self.aname[0]
        self.atom_tensor = np.asarray(self.aname)
        return (self.dimeraname, self.xyz, self.elec, self.exch, self.ind, 
                    self.disp, self.energy, self.atom_tensor, self.en_units)
    
    def grab_xyz_from_psi4out(target_directory,outfile_name):
        with open(outfile_path) as f2:
            count = 0
            for line in f2:
                if "Center" in line:
                    count += 1
                if not line.strip() and count == 1:
                    count -= 1
                if count == 1 and "-" not in line:
                    data = line.split()
                    dimeraname.append(data[0])
                    dimerxyz.append([float(data[1]),float(data[2]),float(data[3])])
            f2.close()
        return self.dimeraname, self.dimerxyz
    
    def read_QM9_data():
        """
        Scrapes all SAPT output files in target_directory 's subdirectories
        Reports molecular geometries and their SAPT energy decomposition labels
        """ 
        for root, dirs, files in os.walk("./%s"%self.target_dir):
            for target in files:
                #for directory_name in dirs:
                geom_file = "./%s/%s"%(self.target_dir,target)
                if os.path.exists(geom_file):
                    file = open(geom_file, "r", errors="ignore")
                    line_num = 0
                    dimeraname = []
                    dimerxyz = []
                    atoms_iterated = 0
                    for line in file:
                        line = line.replace("   "," ")
                        if line_num == 0:
                            num_atoms = int(line.strip())
                            #print(num_atoms)
                        if line_num == 1:
                            data = line.split()
                            energy.append(float(data[12]))
                        if line_num > 1 and atoms_iterated < num_atoms:
                            data = line.split()
                            dimeraname.append(data[0])
                            for k in [1,2,3]:
                                if "*^" in str(data[k]):
                                    data[k] = data[k].replace("*^","e")
                            dimerxyz.append([float(data[1]),float(data[2]),float(data[3])])
                            #print(dimerxyz)
                            atoms_iterated += 1
                        line_num += 1
                    #dimerxyz = np.array(dimerxyz)
                    aname.append(dimeraname)
                    xyz.append(dimerxyz)
                    file.close()
        # change lists to arrays
        self.xyz=np.array(xyz, dtype=object)
        self.energy=np.array(energy)
        self.atom_tensor = self.aname
        return self.dimeraname, self.xyz, self.energy, self.atom_tensor, self.en_units
    
    def read_sapt_data(target_directory):
        """
        Scrapes all SAPT output files in target_directory 's subdirectories
        Reports molecular geometries and their SAPT energy decomposition labels
        """ 
        for root, dirs, files in os.walk("./%s"%target_directory):
            for direc in dirs:
                if direc != 'fsapt':
                    #for directory_name in dirs:
                    geom_file = "./%s/%s/fsapt/geom.xyz"%(target_directory,direc)
                    if os.path.exists(geom_file):
                        file = open(geom_file)
                        line_num = 0
                        dimeraname = []
                        dimerxyz = []
                        for line in file:
                            if line_num > 1:
                                data = line.split()
                                dimeraname.append(data[0])
                                dimerxyz.append([float(data[1]),float(data[2]),float(data[3])])
                            line_num += 1
                        aname.append(dimeraname)
                        xyz.append(dimerxyz)
                        file.close()
                        for rd,drex,filez in os.walk("./%s/%s"%(target_directory,direc)):
                            for outfile in filez:
                                if ".out" in outfile and ".swp" not in outfile:
                                    file = open("./%s/%s/%s"%(target_directory,direc,outfile))
                                    count = 0
                                    for line in file:
                                        if count > 0:
                                            data = line.split()
                                            if count == 3:
                                                val = float(data[3])
                                                E_elec = val
                                            if count == 6:
                                                val = float(data[3])
                                                E_exch = val
                                            if count == 10:
                                                val = float(data[3])
                                                E_ind = val
                                            if count == 17:
                                                val = float(data[3])
                                                E_disp = val
                                            count += 1
                                        
                                        if "SAPT Results" in line:
                                            count += 1
                                        
                                    file.close()
                                    elec.append(E_elec)
                                    ind.append(E_ind)
                                    disp.append(E_disp)
                                    exch.append(E_exch)
                                    etot = E_elec + E_exch + E_ind + E_disp
                                    energy.append(etot)
    
        # change lists to arrays
        self.xyz=np.array(xyz)
        self.elec = np.array(elec)
        self.exch = np.array(exch)
        self.ind = np.array(ind)
        self.disp = np.array(disp)
        self.energy=np.array(energy)
        self.dimername = aname[0]
        self.atom_tensor = np.asarray(aname)
        return (self.aname, self.xyz, self.elec, self.exch, self.ind, self.disp,
                self.energy, self.atom_tensor, self.en_units)
    
    
    def read_sapt_data():
        """
        Scrapes all SAPT output files in target_directory 's subdirectories
        Reports molecular geometries and their SAPT energy decomposition labels
        """ 
        for root, dirs, files in os.walk("./%s"%self.target_dir):
            for direc in dirs:
                if direc != 'fsapt':
                    #for directory_name in dirs:
                    geom_file = "./%s/%s/fsapt/geom.xyz"%(self.target_dir,direc)
                    if os.path.exists(geom_file):
                        file = open(geom_file)
                        line_num = 0
                        dimeraname = []
                        dimerxyz = []
                        for line in file:
                            if line_num > 1:
                                data = line.split()
                                dimeraname.append(data[0])
                                dimerxyz.append([float(data[1]),float(data[2]),float(data[3])])
                            line_num += 1
                        aname.append(dimeraname)
                        xyz.append(dimerxyz)
                        file.close()
                        for rd,drex,filez in os.walk("./%s/%s"%(self.target_dir,direc)):
                            for outfile in filez:
                                if ".out" in outfile and ".swp" not in outfile:
                                    file = open("./%s/%s/%s"%(self.target_dir,direc,outfile))
                                    count = 0
                                    for line in file:
                                        if count > 0:
                                            data = line.split()
                                            if count == 3:
                                                val = float(data[3])
                                                E_elec = val
                                            if count == 6:
                                                val = float(data[3])
                                                E_exch = val
                                            if count == 10:
                                                val = float(data[3])
                                                E_ind = val
                                            if count == 17:
                                                val = float(data[3])
                                                E_disp = val
                                            count += 1
                                        
                                        if "SAPT Results" in line:
                                            count += 1
                                        
                                    file.close()
                                    elec.append(E_elec)
                                    ind.append(E_ind)
                                    disp.append(E_disp)
                                    exch.append(E_exch)
                                    etot = E_elec + E_exch + E_ind + E_disp
                                    energy.append(etot)
    
        # change lists to arrays
        self.xyz=np.array(xyz)
        self.elec = np.array(elec)
        self.exch = np.array(exch)
        self.ind = np.array(ind)
        self.disp = np.array(disp)
        self.energy=np.array(energy)
        self.dimername = aname[0]
        self.atom_tensor = np.asarray(aname)
        return (self.aname, self.xyz, self.elec, self.exch, self.ind, self.disp, 
                self.energy, self.atom_tensor, self.en_units)

    def read_from_target_list(self):
        target_list = self.target_dir
        with open(target_list, "r") as file:
            if self.mode == "alex": next(file)
            for line in file:
                molec = line.split(",")
                filename = molec[0]

                if self.mode == "alex":
                    set_code = molec[8]
                    if "Step" in set_code:
                        train_data = True
                    elif "All_conf" in set_code:
                        train_data = False
                system_name = molec[0].split("_")[0] + "_" + molec[0].split("_")[1]
                if self.mode != "alex":
                    geom_file = "./crystallographic_data/2019-01-07-CSD-NMA-NMA-xyz-nrgs-outs/XYZ/%s.xyz"%(filename)
                else:
                    geom_file = "./XYZ-FILES/%s-xyzfiles/%s.xyz"%(system_name,filename.replace(".trunc.out",""))
                if os.path.exists(geom_file):
                    geom = open(geom_file,"r")
                    line_num = 0
                    dimeraname = []
                    dimerxyz = []
                    atoms_iterated = 0
                    for line in geom:
                        if line_num == 0:
                            num_atoms = int(line.strip())
                        elif line_num == 1:
                            line_num +=1
                            continue
                        elif line_num > 1 and atoms_iterated < num_atoms:
                            data = line.split()
                            dimeraname.append(data[0])
                            dimerxyz.append([float(data[1]),float(data[2]),float(data[3])])
                            atoms_iterated += 1
                        line_num += 1
                    if self.mode == "alex":
                        if train_data:
                            self.training_aname.append(dimeraname)
                            self.training_xyz.append(dimerxyz)
                            self.training_geom_files.append(filename)
                        else:
                            self.testing_aname.append(dimeraname)
                            self.testing_xyz.append(dimerxyz)
                            self.testing_geom_files.append(filename)
                    else:        
                        self.aname.append(dimeraname)
                        self.xyz.append(dimerxyz)
                        self.geom_files.append(filename)
                    geom.close()
                
                else:
                    print("Invalid geometry file path encountered")
                    print(geom_file)
                
                if self.mode != "alex":
                    self.energy.append(molec[1])
                    self.elec.append(molec[2])
                    self.exch.append(molec[3])
                    self.ind.append(molec[4])
                    self.disp.append(molec[5])
                    self.atom_tensor = self.aname
                
                if self.mode == "alex":
                    if train_data == False:
                        self.testing_energy.append(molec[3])
                        self.testing_elec.append(molec[4])
                        self.testing_exch.append(molec[5])
                        self.testing_ind.append(molec[6])
                        self.testing_disp.append(molec[7])
                        self.testing_atom_tensor = self.testing_aname
                    elif train_data == True:
                        self.training_energy.append(molec[3])
                        self.training_elec.append(molec[4])
                        self.training_exch.append(molec[5])
                        self.training_ind.append(molec[6])
                        self.training_disp.append(molec[7])
                        self.training_atom_tensor = self.training_aname
        
        if self.mode != "alex":
            return (self.aname, 
                    self.atom_tensor,
                    self.xyz,
                    self.elec,
                    self.ind, 
                    self.disp,
                    self.exch,
                    self.energy,
                    self.geom_files)
        elif self.mode == "alex":
            return (self.testing_aname, 
                    self.testing_atom_tensor,
                    self.testing_xyz,
                    self.testing_elec,
                    self.testing_ind, 
                    self.testing_disp,
                    self.testing_exch,
                    self.testing_energy,
                    self.testing_geom_files,
                    self.training_aname, 
                    self.training_atom_tensor,
                    self.training_xyz,
                    self.training_elec,
                    self.training_ind, 
                    self.training_disp,
                    self.training_exch,
                    self.training_energy,
                    self.training_geom_files)

def progress(count,total,suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len*count / float(total)))
    percents = round(100.0 * count / float(total),1)
    bar = '=' * filled_len + '-' * (bar_len-filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def infer_on_test(model, sym_inp):
    (en_pred, elec_pred, exch_pred, ind_pred, disp_pred) = model.predict_on_batch(sym_inp)
    return en_pred, elec_pred, exch_pred, ind_pred, disp_pred

def error_statistics(predicted,actual):
    error = np.zeros(len(actual))
    error_sq = np.zeros(len(actual))
    predicted = np.asarray(predicted).astype(float)
    actual = actual.astype(float)
    for i in range(len(actual)):
        error[i] = np.absolute(actual[i] - predicted[i])
        error_sq[i] = (actual[i]-predicted[i])**2
    mae = np.sum(error)/len(actual)
    rmse = np.sqrt(np.sum(error_sq)/len(actual))
    max_error = np.amax(error)
    return mae,rmse,max_error


def get_atomic_num_tensor(atom_tensor, atom_dict):
    atomic_num_tensor = []
    for i in range(len(atom_tensor)):
        molec_atom_num_list = []
        for j in range(len(atom_tensor[i])):
            molec_atom_num_list.append(atom_dict[atom_tensor[i][j]])
        atomic_num_tensor.append(molec_atom_num_list)
    return atomic_num_tensor

def set_modeller_coordinates(pdb, modeller, xyz_in):
    """
    this sets the coordinates of the OpenMM 'pdb' and 'modeller' objects
    with the input coordinate array 'xyz_in'
    """
    # loop over atoms
    for i in range(len(xyz_in)):
        # update pdb positions
        pdb.positions[i]=Vec3(xyz_in[i][0] , xyz_in[i][1], xyz_in[i][2])*nanometer

    # now update positions in modeller object
    modeller_out = Modeller(pdb.topology, pdb.positions)
  
    return modeller_out

def slice_arrays(arrays, start=None, stop=None):
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
        elif hasattr(start,'__getitem__'):
            return arrays[start:stop]
        else:
            return [None]

def create_atype_list(aname, atomic_dict):
    """ 
    creates a lookup list of unique element indices
    atype should be a lookup list of numeric indices
    aname is the tensor of strings of atomic identifiers like 'C' or 'H'   
    atypename should be a lookup list of actual atom names (same as aname..?)
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
    print(unique_list)
    ntype = len(unique_list)
    return ntype, atype 

def compute_displacements(xyz,path):
    """
    computes displacement tensor from input cartesian coordinates
    """
    Parallel(n_jobs=4, verbose=1)(delayed(displacement_matrix)(xyz[i],path,i) for i in range(len(xyz)))
    #for i in range(len(xyz)):
    #    displacement_matrix(xyz[i],path,i)
    return

def displacement_matrix(xyz,path,i):
        molec_rij = []
        molec_dr = []
        rij_out = "%s/rij_%s.npy"%(path,i)
        dr_out = "%s/dr_%s.npy"%(path,i)
        # first loop over atoms
        for i_atom in range(len(xyz)):
            atom_rij = []
            atom_dr = []
            # second loop over atoms
            for j_atom in range(len(xyz)):
                x_diff = xyz[i_atom][0] - xyz[j_atom][0]
                y_diff = xyz[i_atom][1] - xyz[j_atom][1]
                z_diff = xyz[i_atom][2] - xyz[j_atom][2]
                atom_dr.append([x_diff,y_diff,z_diff])
                atom_rij.append(LA.norm([x_diff,y_diff,z_diff]))
            molec_dr.append(atom_dr)
            molec_rij.append(atom_rij)
        #rij.append(molec_rij)
        #dr.append(molec_dr)
        np.save(rij_out, np.array(molec_rij))
        np.save(dr_out, np.array(molec_dr))


def compute_thetas(num_systems,path):
    """
    computes theta tensor from displacement tensor
    """
    Parallel(n_jobs=4, verbose=1)(delayed(theta_tensor_single)(path,i) for i in range(num_systems))
    #for i in range(num_systems):
    #    theta_tensor_single(path,i) 
    return

def theta_tensor_single(path,i): 
    molec_theta = []
    outfile = "%s/theta_%s.npy"%(path,i)
    rij = np.load("%s/rij_%s.npy"%(path,i))
    dr = np.load("%s/dr_%s.npy"%(path,i))
    for i_atom in range(len(rij)):
        atom_theta = []
        for j_atom in range(len(rij)):
            atom_atom_theta = []
            for k_atom in range(len(rij)):
                r1 = rij[i_atom][j_atom]
                r2 = rij[i_atom][k_atom]
                dr1 = dr[i_atom][j_atom]
                dr2 = dr[i_atom][k_atom]
                if i_atom!=j_atom and i_atom!=k_atom and j_atom!=k_atom:
                    ph_angle = np.dot(dr1,dr2)/r1/r2
                    if ph_angle>1: ph_angle=1
                    if ph_angle<-1: ph_angle=-1
                    atom_atom_theta.append(np.arccos(ph_angle))
                else:
                    atom_atom_theta.append(float(0.0))
                    
            atom_theta.append(atom_atom_theta)
        molec_theta.append(atom_theta)
    np.save(outfile, np.array(molec_theta))
    return

def atomwise_displacements(xyz):
    """
    computes displacement tensor of length 6 vectors from input cartesian coordinates...
    makes rii = 99.9 in order to override cutoff distance and remove self-contribution
    """

    rij=np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[1]))
    for i in range(len(xyz)):
        for i_atom in range(len(xyz[0])):
            for j_atom in range(len(xyz[0])):
                if i_atom == j_atom:
                    rij[i][i_atom][j_atom] = 99.9
                else:
                    dr = xyz[i][i_atom] - xyz[i][j_atom]
                    rij[i][i_atom][j_atom] = LA.norm(dr)
    return rij 
      
def flat_displacements(xyz):
    """
    computes a flattened vector of displacements for every data point
    """
    rij = np.zeros((xyz.shape[0],xyz.shape[1]**2))
    for i in range(len(xyz)):
        for i_atom in range(len(xyz[0])):
            for j_atom in range(len(xyz[0])):
                flat_index = j_atom + i_atom*len(xyz[0])
                dr = xyz[i][i_atom] - xyz[i][j_atom]
                rij[i][flat_index] = LA.norm(dr)
    return rij

def construct_symmetry_input(NN, path, aname , ffenergy, atomic_num_tensor, val_split):
    """
    this subroutine constructs input tensor for NN built from
    elemental symmetry functions
 
    NOTE: We assume we are using a shared layer NN structure, where
    layers are shared for common atomtypes (elements).  We then will concatenate
    networks into single layers. For this structure, we need input data to be a list of
    arrays, where each array is the input data (data points, symmetry functions) for each
    atom.

    Thus, the 'sym_input' output of this subroutine is a list of 2D arrays, of form
    [ data_atom1 , data_atom2 , data_atom3 , etc. ] , and data_atomi is an array [ j_data_point , k_symmetry_function ]
    """
    sym_path = path.replace("spatial_info","sym_inp")
    os.mkdir(sym_path)
    Parallel(n_jobs=4, verbose=1)(delayed(sym_inp_single)(sym_path,path,j_molec,
                aname,NN,atomic_num_tensor[j_molec]) for j_molec in range(len(aname)))
    #sym_input = []
    #for j_molec in range(len(aname)): #loop over all molecules in dataset
    #    input_atom = sym_inp_single(path,j_molec,aname,NN,atomic_num_tensor[j_molec])
    #    sym_input.append(input_atom)
    return
        
def sym_inp_single(sym_path,path,j_molec,aname,NN,atomic_num_slice):
    outfile = "%s/symfun_%s.npy"%(sym_path,j_molec)
    rij = np.load("%s/rij_%s.npy"%(path,j_molec))
    theta = np.load("%s/theta_%s.npy"%(path,j_molec))
    input_atom = []
    
    for i_atom in range(len(aname[j_molec])): #loop over all atoms in each molec
        typeNN = NN.element_force_field[aname[j_molec][i_atom]]
        input_i = []
        for i_sym in range(len(typeNN.radial_symmetry_functions)):
                #looping over symfuns for this atom
            G1 = sym.radial_gaussian(rij, i_atom, 
                typeNN.radial_symmetry_functions[i_sym][0],
                typeNN.radial_symmetry_functions[i_sym][1],
                typeNN.Rc, atomic_num_slice)
            input_i.append(G1)

        for i_sym in range(len(typeNN.angular_symmetry_functions)):
            G2 = sym.angular_gaussian(rij, theta, i_atom,
                typeNN.angular_symmetry_functions[i_sym][0],
                typeNN.angular_symmetry_functions[i_sym][1],
                typeNN.angular_symmetry_functions[i_sym][2],
                typeNN.Rc, atomic_num_slice)
            input_i.append(G2)
        input_atom.append(input_i)
    np.save(outfile, np.array(input_atom)) #only supports when all atoms have same # of features
    return

def construct_flat_symmetry_input(NN , rij, aname , ffenergy):
    """
    This flat version can pretty much only be used when the entire data set has the same # of atoms and atomtypes...
    """
    # output tensor with symmetry function input
    flat_sym_input=[]

    # loop over data points
    for i in range( rij.shape[0] ):
        # loop over atoms
        input_i = [ffenergy[i]]
        for i_atom in range( len(aname) ):
            # get elementNN object for this atomtype
            typeNN = NN.element_force_field[ aname[i_atom] ]

            # now loop over symmetry functions for this element
            for i_sym in range( len( typeNN.symmetry_functions ) ):
                # construct radial gaussian symmetry function
                G1 = sym.radial_gaussian( rij[i] , i_atom , typeNN.symmetry_functions[i_sym][0] , typeNN.symmetry_functions[i_sym][1], typeNN.Rc )
                input_i.append( G1 )
        # convert 2D list to array
        input_i = np.array( input_i )
        # append to list
        flat_sym_input.append(input_i)
    flat_sym_input = np.array(flat_sym_input)
    return flat_sym_input

class SymLayer(Layer):
    """
    Under Construction: this is to be used as the first layer of a NN that takes rij and associated atomtypes as input
    """
    def __init__(self,**kwargs):
        super(SymLayer,self).__init__(**kwargs)

    def compute_output_shape(self,input_shape):
        shape = 1   #again, temporary to avoid passing in a bunch of junk for now
        return shape

    def build(self,input_shape):
        self.width_init = K.ones(1)
        self.shift_init = K.ones(1)
        self.width = K.variable(self.width_init,name="width",dtype='float32')
        self.shift = K.variable(self.shift_init,name="shift",dtype='float32')
        self.trainable_weights = [self.width, self.shift]
        super(SymLayer,self).build(input_shape)
        
    def call(self,x):
        Rc = 12.0
        PI = 3.14159265359
        # loop over num. radial sym functions requested
        #for i in range(5):
        # loop over atoms in molec
        G = K.zeros(1,dtype='float32')
        self.width = tf.cast(self.width,tf.float32)
        for j_atom in range( 6 ): #this is super specific for water dimer, fixable
            print(x)
            fc = tf.cond(x<=Rc, lambda: 0.5*(K.cos(PI*x)+1), lambda: 0.0)
            G = G + fc * K.exp(-self.width * (K.pow((x[j_atom] - self.width),2)))
            #K.concatenate([G],axis=0)
        print(G)
        return G


def custom_loss(y_true, y_pred):
    total_en_loss = K.mean(K.square(y_true[0]-y_pred[0]))
    elst_loss = K.mean(K.square(y_true[1]-y_pred[1]))
    exch_loss = K.mean(K.square(y_true[2]-y_pred[2]))
    ind_loss = K.mean(K.square(y_true[3]-y_pred[3]))
    disp_loss = K.mean(K.square(y_true[4]-y_pred[4]))
    comp_weight = 0
    loss = (1-comp_weight)*total_en_loss + comp_weight*(elst_loss + exch_loss + ind_loss + disp_loss)
    loss = total_en_loss
    return loss

def scale_symmetry_input( sym_input ):
    """
    this subroutine scales the symmetry input to lie within the range [-1,1]
    """
    small=1E-6
    # loop over atoms
    for i_atom in range( len(sym_input) ):
        # loop over symmetry functions
        for i_sym in range( sym_input[i_atom].shape[1] ):
            # min and max values of this symmetry function across the data set
            min_sym = np.amin(sym_input[i_atom][:,i_sym] )
            max_sym = np.amax(sym_input[i_atom][:,i_sym] )
            range_sym = max_sym - min_sym

            if range_sym > small:
                for i_data in range( sym_input[i_atom].shape[0] ):
                    sym_input[i_atom][i_data,i_sym] = -1.0 + 2.0 * ( ( sym_input[i_atom][i_data,i_sym] - min_sym ) / range_sym )
            
            # shift symmetry function value so that histogram maximum is centered at 0
            # this is for increasing sensitivity to tanh activation function
            hist = np.histogram( sym_input[i_atom][:,i_sym] , bins=20 )
            shift = np.argmax(hist[0])
            array_shift = np.full_like( sym_input[i_atom][:,i_sym] , -1.0 + 2*( shift / 20 ) )
            sym_input[i_atom][:,i_sym] = sym_input[i_atom][:,i_sym] - array_shift[:]
    return sym_input

def test_symmetry_input(sym_input):
    """
    for a symmetry function to contribute to a neural network
    it must exhibit a non-negligable variance across the data set
    here we test the variance of symmetry functions across data set
    """

    # loop over atoms
    for i_atom in range( len(sym_input) ):
        print ("symmetry functions for atom ", i_atom )
        
        # histogram values of every symmetry function
        for i_sym in range( sym_input[i_atom].shape[1] ):
  
            hist = np.histogram( sym_input[i_atom][:,i_sym] )

            for i in range(len(hist[0])):
                print( hist[1][i] , hist[0][i] )
            print( "" )

    sys.exit()

def atomic_dictionary():
    atom_dict = {
        'H'  : 1,
        'He' : 2,
        'Li' : 3,
        'Be' : 4,
        'B' : 5,
        'C' : 6,
        'N' : 7,
        'O' : 8,
        'F' : 9,
        'Ne' : 10,
        'Na' : 11,
        'Mg' : 12,
        'Al' : 13,
        'Si' : 14,
        'P' : 15,
        'S' : 16,
        'Cl' : 17,
        'Ar' :18,
        'K' : 19,
        'Ca' : 20,
        'Sc' : 21,
        'Ti' : 22,
        'V' :  23,
        'Cr' : 24,
        'Mn' : 25,
        'Fe' : 26,
        'Co' : 27,
        'Ni' : 28,
        'Cu' : 29,
        'Zn' : 30,
        'Ga' : 31,
        'Ge' : 32,
        'As' : 33,
        'Se' : 34,
        'Br' : 35,
        'Kr' : 36,
        'Rb' : 37,
        'Sr' : 38,
        'Y' : 39,
        'Zr' : 40,
        'Nb' : 41,
        'Mo' : 42,
        'Tc' : 43,
        'Ru' : 44,
        'Rh' : 45,
        'Pd' : 46,
        'Ag' : 47,
        'Cd' : 48,
        'In' : 49,
        'Sn' : 50,
        'Sb' : 51,
        'Te' : 52,
        'I' : 53,
        'Xe' : 54
    }
    return atom_dict
