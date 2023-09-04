#resim.py

import argparse
import os
import pickle
import shutil 
import meep as mp
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from IPython import embed
import sys

sys.path.append('../')
from utils import mapping, parameter_manager

if __name__=="__main__":
    
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.ParameterManager(params=params)
    path_results = "/develop/results/spie_journal_2023"

    dataset = "train"
    idx = int(0)

    path_resims = os.path.join(path_results, pm.exp_name + '_2', dataset + '_info') 
    model_results = pickle.load(open(os.path.join(path_resims,'resim.pkl'), 'rb'))

    phases = model_results['phase_pred'][idx]

    radii_list = mapping.phase_to_radii(phases)
    radii_list = np.round(radii_list, 4)
    r0 = float(radii_list[0])
    r1 = float(radii_list[1])
    r2 = float(radii_list[2])
    r3 = float(radii_list[3])
    r4 = float(radii_list[4])
    r5 = float(radii_list[5])
    r6 = float(radii_list[6])
    r7 = float(radii_list[7])
    r8 = float(radii_list[8])
    sys.path.append('/develop/code/surrogate_model/src/3x3_pillar_sims')
    cmd = f'mpirun -np 2 python3 run_sim.py -r0 {r0} -r1 {r1} -r2 {r2} -r3 {r3} -r4 {r4} -r5 {r5} -r6 {r6} -r7 {r7} -r8 {r8} -resim 1 -index {idx} -exp_name {pm.exp_name} -dataset {dataset}'
    #cmd = f"mpirun --allow-run-as-root -np 32 python3 run_sim.py -r0 {r0} -r1 {r1} -r2 {r2} -r3 {r3} -r4 {r4} -r5 {r5} -r6 {r6} -r7 {r7} -r8 {r8} -index {i}, -folder_name {folder_name}"
    os.system(cmd)
