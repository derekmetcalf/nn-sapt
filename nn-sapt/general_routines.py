import os
import sys
import numpy as np
from numpy import linalg as LA

def read_sapt_data(target_directory):
    """
    scrapes all SAPT output files in target_directory 's subdirectories
    returns molecular geometries and their SAPT energy decomposition labels
    """ 
    aname = []
    xyz= []
    elec = []
    ind = []
    disp = []
    exch = []
    delHF = []
    energy = []
    en_units = "kcal/mol"
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
    xyz=np.array(xyz)
    elec = np.array(elec)
    exch = np.array(exch)
    ind = np.array(ind)
    disp = np.array(disp)
    energy=np.array(energy)
    dimername = aname[0]
    atom_tensor = np.asarray(aname)
    return dimeraname, xyz, elec, exch, ind, disp,  energy, atom_tensor, en_units


def error_statistics(predicted,actual):
    error = np.zeros(len(actual))
    error_sq = np.zeros(len(actual))
    for i in range(len(actual)):
        error[i] = np.absolute(actual[i] - predicted[i])
        error_sq[i] = (actual[i]-predicted[i])**2
    mae = np.sum(error)/len(actual)
    rmse = np.sqrt(np.sum(error_sq)/len(actual))
    max_error = np.amax(error)
    return mae,rmse,max_error


def get_atomic_num_tensor(atom_tensor, atom_dict, atype):
    atomic_num_tensor = np.zeros(np.shape(atom_tensor))
    print(atom_tensor[2])
    print(np.shape(atom_tensor))
    for i in range(len(atom_tensor)):
        for j in range(len(atom_tensor[0])):
            atomic_num_tensor[i,j] = atom_dict[atom_tensor[i,j]] 
    return atomic_num_tensor

def create_atype_list(aname, atomic_dict):
    """ 
    creates a lookup list of unique element indices
    """
    index=0
    # fill in first type
    atype=[]
    atypename=[]
    atype.append(index)
    atypename.append(aname[index])  

    for i in range(1, len(aname)):
        # find first match
        flag=-1
        for j in range(i):
            if aname[i] == aname[j]:
               flag=atype[j]
        if flag == -1:
            index+=1
            atype.append(index)
            atypename.append(aname[i])
        else:
            atype.append( flag )
    atomic_number=[]
    for i in range(len(atypename)):
        atomic_number.append(atomic_dict.get(atypename[i],'none'))
    ntype = index + 1
    return ntype, atype, atypename, atomic_number 

def compute_displacements(xyz):
    """
    computes displacement tensor from input cartesian coordinates
    """
    rij = np.zeros((xyz.shape[0],xyz.shape[1],xyz.shape[1])) # tensor of pairwise distances for every data point
    dr = np.zeros((xyz.shape[0],xyz.shape[1],xyz.shape[1],3))
    # loop over data points
    for i in range(len(xyz)):
        # first loop over atoms
        for i_atom in range(len(xyz[0])):
            # second loop over atoms
            for j_atom in range(len(xyz[0])):
                dr[i][i_atom][j_atom] = xyz[i][i_atom] - xyz[i][j_atom]
                rij[i][i_atom][j_atom] = LA.norm(dr[i][i_atom][j_atom])
    return rij,dr

def compute_thetas(rij,dr):
    """
    computes theta tensor from displacement tensor
    """
    theta = np.zeros((rij.shape[0],rij.shape[1],rij.shape[1],rij.shape[1]))
    for i in range(len(rij)):
        for i_atom in range(len(rij[0])):
            for j_atom in range(len(rij[0])):
                for k_atom in range(len(rij[0])):
                    r1 = rij[i][i_atom][j_atom]
                    r2 = rij[i][i_atom][k_atom]
                    dr1 = dr[i][i_atom][j_atom]
                    dr2 = dr[i][i_atom][k_atom]
                    if i_atom!=j_atom and i_atom!=k_atom and j_atom!=k_atom:
                        theta[i][i_atom][j_atom][k_atom] = np.arccos(np.dot(dr1,dr2)/r1/r2)
                    else:
                        theta[i][i_atom][j_atom][k_atom] = float('nan')
    return theta

def atomwise_displacements( xyz ):
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
      
def flat_displacements( xyz ):
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
