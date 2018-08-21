from keras.models import load_model
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from force_field_parameters import *
import matplotlib.pyplot as plt
import routines3 as routines
import numpy as np
import FFenergy_openMM as saptff
import time

inputfile = "h2o_h2o_dummy_Eint_10mh.bohr"
pdb = PDBFile("h2o_template.pdb")
(aname, xyz, energy) = routines.read_sapt_data2(inputfile)
(energy, ffenergy, residual) = saptff.resid(inputfile, pdb)
NNff = NNforce_field('FF1',0,0)
rij, dr = routines.compute_displacements(xyz)
sym_input = routines.construct_symmetry_input(NNff, rij, aname, ffenergy, val_split=0)
#sym_input = np.asarray(sym_input).transpose(1,0,2)
model = load_model("model_test.h5")

print("Beginning predictions:\n")
T = 200
prediction = np.zeros(np.size(energy))
variance = np.zeros(np.size(energy))
abs_error = np.zeros(np.size(energy))
start = time.time()
for i in range(len(energy)):
    preds = np.zeros(T)
    symfuns=[]
    for k in range(len(np.asarray(sym_input)[:,i,:])):
        symfuns.append(np.asarray(sym_input)[k,i:(i+1),:])
        #print(symfuns)
    for j in range(T):
        preds[j] = model.predict(symfuns)
    prediction[i] = np.mean(preds)
    variance[i] = np.var(preds)
    abs_error[i] = np.abs(residual[i] - prediction[i])
end = time.time()
print("Time predicting: %s\n"%(end-start))
print("Time per system: %s"%((end-start)/len(energy)))
print("MAE = %s"%(np.mean(abs_error)))
#plt.scatter(prediction, energy, s=0.9, color="xkcd:red")
#plt.show()
"""
plt.scatter(residual, prediction, s=0.9, color="xkcd:red")
plt.xlabel("True SAPT(DFT)-SAPTFF Residual")
plt.ylabel("NN-Predicted Residuals")
plt.show()
"""
plt.scatter(residual, prediction, s=0.9, color="xkcd:red")
plt.xlabel("True SAPT(DFT)-SAPTFF Residual")
plt.ylabel("NN-Predicted Residuals")
plt.show()

line = np.linspace(-30,30)
plt.subplot(2,1,1)
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.plot(line,line,color='xkcd:black')
plt.scatter(energy, ffenergy+prediction, s=0.9, color='xkcd:red')
plt.ylabel('NN + FF energy (kJ/mol)')

plt.subplot(2,1,2)
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.plot(line,line,color='xkcd:black')
plt.scatter(ffenergy, energy, s=0.9, color='xkcd:red')
plt.xlabel('SAPT energy (kJ/mol)')
plt.ylabel('FF energy (kJ/mol)')
plt.show()

plt.scatter(variance, np.abs(energy - (ffenergy+prediction)), s=0.9, color="xkcd:red")
plt.xlabel("Neural Network Prediction Variance")
plt.ylabel("NN Error Compared to SAPT(DFT)-SAPTFF Residual Labels")
plt.show()
