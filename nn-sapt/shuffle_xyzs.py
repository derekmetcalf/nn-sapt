import numpy as np
import routines
import symfun_parameters

path = "./shuffled_xyzs"

NN = symfun_parameters.NNforce_field("GA_opt",0,0)
(atoms,atom_nums,xyz) = routines.get_xyz_from_combo_files(path)
(filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)

def shuffle(xyz,atoms,atom_nums):
    shuffled_xyz = np.zeros(np.shape(xyz))
    shuffled_atoms = np.zeros(np.shape(atoms),dtype=object)
    shuffled_atom_nums = np.zeros(np.shape(atom_nums))
    for i in range(len(xyz)):
        permutation = np.random.permutation(len(xyz[i]))
        for j in permutation:
            for k in range(len(xyz[i])):
                shuffled_xyz[i][k] = xyz[i][j]
                shuffled_atoms[i][k] = atoms[i][j]
                shuffled_atom_nums[i][k] = atom_nums[i][j]
    return shuffled_xyz,shuffled_atoms,shuffled_atom_nums

(xyz,atoms,atom_nums) = shuffle(xyz,atoms,atom_nums)

routines.compute_displacements(xyz,path,filenames)
routines.compute_thetas(len(xyz),path,filenames)
routines.construct_symmetry_input(NN,path,filenames,atoms,
                            np.zeros(len(xyz)),atom_nums,0)


