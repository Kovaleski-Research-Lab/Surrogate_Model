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
import run_sim

sys.path.append('../')
from utils import mapping, parameter_manager

if __name__=="__main__":
   
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.ParameterManager(params=params)
    pm.resim = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("-index", type=int, help="Index - neighbor index if generating data, if doing resim, a value between 0 and 8") 
    parser.add_argument("-dataset", type=str, help="Train or valid - if we're doing resims")
    
    args = parser.parse_args()
    idx = args.index
    dataset = args.dataset

    path_results = "/develop/results/spie_journal_2023"

    # Need to get phase values from the model's predictions.
    path_resims = os.path.join(path_results, pm.exp_name + '_2', dataset + '_info') # might need to change this too 
    model_results = pickle.load(open(os.path.join(path_resims,'resim.pkl'), 'rb'))

    phases = model_results['phase_pred'][idx]

    radii_list = mapping.phase_to_radii(phases)
    radii_list = np.round(radii_list, 6)
    run_sim.run(radii_list, idx, pm, dataset) # idx identifies the index of results we're getting a resim for
    
