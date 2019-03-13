import os
import sys
import numpy as np
import time
from numpy import linalg as LA
import symmetry_functions as sym
from symfun_parameters import *
import routines

"""Combine and handle desired data subsets."""

all_paths = []
with open(
        "./NMe-acetamide_Indazole/NMe-acetamide_Indazole-review/NMe-acetamide_Indazole.NRGs.txt"
) as f1:
    for line in f1:
        all_paths.append(line.split()[0])
f1.close()

subset_names = []
with open("NMe-acetamide_Indazole_SAPT0-NRGs-kcal-d2.csv") as f2:
    for line in f2:
        subset_names.append(line.split(",")[0])
f2.close()

relevant_dir = []
for pathname in all_paths:
    for filename in subset_names:
        if filename in pathname:
            relevant_dir.append(pathname.split("/")[0])

for direc in relevant_dir:
    os.system("mv ./NMe-acetamide_Indazole/%s ./challenge_system" % direc)
