import routines

path = "./test_combo_xyz_path"

(atoms,xyz) = routines.get_xyz_from_combo_files(path)


(filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)

routines.make_perturbed_xyzs(path)