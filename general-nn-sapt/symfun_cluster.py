from __future__ import print_function
import os
import csv
import sys
import math
import time
import routines
import itertools
import matplotlib.pyplot as plt
import numpy as np
import FFenergy_openMM as saptff
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from symfun_parameters import *

"""Attempt to cluster symmetry functions and do PCA on them.

This file is a curiosity study to see if there is worthwhile information
contained in the clustering of atoms in the symmetry function input 
space. Positive outcomes from this might have active learning implications.

"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = './SSI_spiked'

aname = []
atom_tensor = []
xyz = []
elec = []
ind = []
disp = []
exch = []
energy = []
geom_files = []

sym_input = []
(atoms,atom_nums,xyz) = routines.get_xyz_from_combo_files(path)
(filenames,tot_en,elst,exch,ind,disp,val_split) = routines.get_sapt_from_combo_files(path)

symfun_files = []
for j in range(len(filenames)):
    file = f"{path}/{filenames[j]}_symfun.npy"
    symfun_files.append(file)
    sym_input.append(np.load(file))
"""
sym_input = np.transpose(sym_input, (1, 0, 2))
print(np.shape(atom_tensor))
print(np.shape(sym_input))
X_extra, sym_input, extra_atoms, atom_tensor = train_test_split(
    sym_input, atom_tensor, test_size=0.05)
X_extra = np.transpose(X_extra, (1, 0, 2))
sym_input = np.transpose(sym_input, (1, 0, 2))
"""
flat_atoms = []
flat_sym_inp = []
for i in range(len(atoms)):
    for j in range(len(atoms[i])):
        flat_atoms.append(atoms[i][j])
        flat_sym_inp.append(sym_input[i][j])
flat_atoms = np.array(flat_atoms)
flat_sym_inp = np.array(flat_sym_inp)
"""print(num_inps)
atoms = np.reshape(
    atoms, (num_inps),
    order='F')
sym_input = np.reshape(
    sym_input,
    (np.shape(sym_input)[0] * np.shape(sym_input)[1], np.shape(sym_input)[2]))
"""

pca = PCA(n_components=3)
pca.fit(flat_sym_inp)

sym_input = pca.transform(flat_sym_inp)

H_tensor = []
C_tensor = []
N_tensor = []
O_tensor = []
F_tensor = []
S_tensor = []
F_count = 0
S_count = 0
H_count = 0
C_count = 0
N_count = 0
O_count = 0


for i in range(len(sym_input)):
    if flat_atoms[i] == "H":
        H_tensor.append(sym_input[i, :])
        H_count += 1
    if flat_atoms[i] == "C":
        C_tensor.append(sym_input[i, :])
        C_count += 1
    if flat_atoms[i] == "N":
        N_tensor.append(sym_input[i, :])
        N_count += 1
    if flat_atoms[i] == "O":
        O_tensor.append(sym_input[i, :])
        O_count += 1
    if flat_atoms[i] == "S":
        S_tensor.append(sym_input[i, :])
        S_count += 1
    if flat_atoms[i] == "F":
        F_tensor.append(sym_input[i, :])
        F_count += 1
H_tensor = np.array(H_tensor)
C_tensor = np.array(C_tensor)
N_tensor = np.array(N_tensor)
O_tensor = np.array(O_tensor)
F_tensor = np.array(F_tensor)
S_tensor = np.array(S_tensor)

#n_components=8
#gmm = GaussianMixture(n_components=n_components).fit(sym_input)
atoms = [H_tensor, C_tensor, N_tensor, O_tensor, F_tensor, S_tensor]
labels = ['H', 'C', 'N', 'O', 'F', 'S']
colors = ['b', 'r', 'g', 'c', 'xkcd:orange','xkcd:purple']

total = H_count + C_count + N_count + O_count + F_count + S_count

#for label in range(n_components):
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
k = 0

atoms = [H_tensor]
#cmap = matplotlib.cm.get_cmap('viridis')
#normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
#col = [cmap(normalize(value) for value in z]
for atom in atoms:
    if len(atom) > 0:
        col = colors[k]
        lab = labels[k]
        ax.scatter(atom[:, 0], atom[:, 1], atom[:, 2], s=0.2, 
                   c=col, label=lab)
        
        k += 1
ax.legend()
plt.show()
