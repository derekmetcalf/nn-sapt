import routines

new_path = "../data/NMe-acetamide_Don--Aniline_crystallographic"
energy_file = "../data/crystallographic_data/NMA-Aniline-xyz-nrgs/FSAPT0/SAPT0-NRGS-COMPONENTS.txt"
xyz_path = "../data/crystallographic_data/NMA-Aniline-xyz-nrgs/XYZ"

routines.combine_energies_xyz(energy_file, xyz_path, new_path)

#(atoms,xyz) = routines.get_xyz_from_combo_files(path)

#(filenames,tot_en,elst,exch,ind,disp) = routines.get_sapt_from_combo_files(path)

#routines.make_perturbed_xyzs(path)
