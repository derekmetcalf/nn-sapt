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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
inputdir='challenge_system'

print("Collecting properties and geometries...\n")
t1 = time.time()
NNff = NNforce_field('GA_opt',0,0)
(aname, xyz, elec, exch, ind, disp, energy, atom_tensor, en_units) = routines.read_sapt_data(inputdir)
t2 = time.time()
elapsed = math.floor(t2-t1)
print("Properties and geometries collected in %s seconds\n"%elapsed)
print("Loading symmetry functions from file...\n")
t1 = time.time()
sym_dir = "%s_sym_input"%inputdir
sym_input = []
for i in range(len([name for name in os.listdir("./%s"%sym_dir) if os.path.isfile(os.path.join(sym_dir,name))])):
    sym_input.append(np.load(os.path.join(sym_dir,"%s.npy"%i)))
sym_input = np.array(sym_input)
t2 = time.time()
elapsed = math.floor(t2-t1)
print("Symmetry functions loaded in %s seconds\n"%elapsed)

sym_input = np.transpose(sym_input, (1,0,2))
print(np.shape(atom_tensor))
print(np.shape(sym_input))
X_extra, sym_input, extra_atoms, atom_tensor = train_test_split(sym_input, atom_tensor, test_size=0.05)
X_extra = np.transpose(X_extra, (1,0,2))
sym_input = np.transpose(sym_input, (1,0,2))

sym_input = np.reshape(sym_input, (np.shape(sym_input)[0]*np.shape(sym_input)[1],np.shape(sym_input)[2]))
atom_tensor = np.reshape(atom_tensor, (np.shape(atom_tensor)[0]*np.shape(atom_tensor)[1]),order='F')

pca = PCA(n_components=3)
pca.fit(sym_input)

sym_input = pca.transform(sym_input)
H_tensor = []
C_tensor = []
N_tensor =[]
O_tensor = []
H_count = 0
C_count = 0
N_count = 0
O_count = 0

for i in range(len(sym_input)): 
    if atom_tensor[i] == "H":
        H_tensor.append(sym_input[i,:]) 
        H_count += 1
    if atom_tensor[i] == "C":
        C_tensor.append(sym_input[i,:]) 
        C_count += 1
    if atom_tensor[i] == "N":
        N_tensor.append(sym_input[i,:]) 
        N_count += 1
    if atom_tensor[i] == "O":
        O_tensor.append(sym_input[i,:]) 
        O_count += 1
H_tensor = np.array(H_tensor)
C_tensor = np.array(C_tensor)
N_tensor =np.array(N_tensor)
O_tensor = np.array(O_tensor)

#n_components=8
#gmm = GaussianMixture(n_components=n_components).fit(sym_input)
atoms = [H_tensor, C_tensor, N_tensor, O_tensor]
labels = ['H', 'C', 'N', 'O']
colors = ['b','r','g','c']

total = H_count + C_count + N_count + O_count

#for label in range(n_components):
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
k = 0
for atom in atoms:
    col = colors[k]
    lab = labels[k]
    ax.scatter(atom[:,0],atom[:,1],atom[:,2],s=0.2, c=col,label=lab)
    k +=1
ax.legend()
plt.show()
