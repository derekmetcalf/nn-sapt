import os
import numpy as np
import routines
import symfun_parameters

#path = "../data/20190413Acc--NMe-acetamide_Don--Aniline-CSD-PLDB-198-dmformat-xyz/Acc--NMe-acetamide_Don--Aniline-PLDB-158-dmformat-xyz"
path = "../data/pert-xyz-nrgs-acceptors-derek-format-test"
NN_A = symfun_parameters.NNforce_field("GA_opt",0,0)
#NN_B = symfun_parameters.NNforce_field("intramolecular",0,0)
#NN_A = symfun_parameters.NNforce_field("intramolecular",0,0)
NN_B = symfun_parameters.NNforce_field("GA_opt",0,0)


folders = []
for r,d,f in os.walk(path):
    for folder in d:
        folders.append(os.path.join(r,folder))
#folders = ["./SSI_neutral/"]
#folders = ["../data/20190413-test-molecules-10k-dm-format/NMA_Quinilone_random"]
#folders = ["test_inter_symfuns"]
systems = 0
for f in folders:
    systems += 1
    if systems > 3:
        print(f"System number {systems}")
        print(f"{f.split('/')[-1]}")
        (filenames,tot_en,elst,exch,ind,disp,split_vec) = routines.get_sapt_from_combo_files(f)
        (atoms,atom_nums,xyz) = routines.get_xyz_from_combo_files(f, filenames)
        #print(split_vec)
        split_vec = np.full((len(xyz)),12)
        routines.compute_displacements(xyz, f, filenames)
        routines.compute_thetas(len(xyz), f, filenames)
        #print(len(atoms))
        
        routines.construct_symmetry_input(NN_A,f,filenames,atoms,np.zeros(len(xyz)), atom_nums,0,NN_B,split_vec)
        
    
print(systems)
