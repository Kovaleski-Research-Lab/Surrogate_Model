import os
import argparse
import pickle
import threading
import numpy as np
from IPython import embed
from numpy.polynomial.polynomial import Polynomial
import torch
import pickle

phase_to_radii_mapper = pickle.load(open("phase_to_radii.pkl","rb"))
#radii_to_phase_mapper = pickle.load(open("radii_to_phase.pkl", "rb"))

radii_list = []
#phase_list = []
path = '/develop/code/cai_2023/marshall'
filename ="metaLens_phaseFunction_f_0.1mm.pkl"
file = os.path.join(path,filename)

phases = pickle.load(open(file,"rb"))
min = -3.00185
max = 2.67187

tphases = torch.tensor(phases)
phases = torch.clamp(tphases, min, max)
phases = phases.numpy()

for phase in phases:
    radii_list.append(Polynomial(phase_to_radii_mapper)(phase))

radii_list = np.asarray(radii_list)

pickle.dump(radii_list, open("radii.pkl","wb"))
