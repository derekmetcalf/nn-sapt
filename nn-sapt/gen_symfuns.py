import numpy as np
import routines
import symfun_parameters

path = "./test_combo_xyz_path"
NN = symfun_parameters.NNforce_field("GA_opt",0,0)
(atoms,atom_nums,xyz) = routines.get_xyz_from_combo_files(path)


(filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)

#routines.compute_displacements(xyz, path, filenames)
#routines.compute_thetas(len(xyz), path, filenames)
routines.construct_symmetry_input(NN, path, filenames, atoms, 
                                    np.zeros(len(xyz)),atom_nums,0)
