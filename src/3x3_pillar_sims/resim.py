import argparse
import os
import pickle
import shutil 

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from IPython import embed
import sys
sys.path.append('/develop/code/')

from utils import mapping


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r0", type=float)
    parser.add_argument("-r1", type=float)
    parser.add_argument("-r2", type=float)
    parser.add_argument("-r3", type=float)
    parser.add_argument("-r4", type=float)
    parser.add_argument("-r5", type=float)
    parser.add_argument("-r6", type=float)
    parser.add_argument("-r7", type=float)
    parser.add_argument("-r8", type=float)
    parser.add_argument("-index", type=int, help="The index matching the index in radii_neighbors")
    parser.add_argument("-folder_name", help="Contains info about the model")
    parser.add_argument("-dataset", help="Train or Valid")
    
    args = parser.parse_args()
    
    idx = args.index
    folder_name = args.folder_name
    dataset = args.dataset

    path_results = "/develop/results/"

    if dataset == 'train':
        path_results = os.path.join(path_results, folder_name, 'train_info')
    elif dataset == 'valid':
        path_results = os.path.join(path_results, folder_name, 'valid_info')
    else:
        exit()

    model_results = pickle.load(open(os.path.join(path_results,'resim.pkl'), 'rb'))

    phases = model_results['phase_pred'][idx]

    radii_list = mapping.phase_to_radii(phases)

#    run(radii_list, folder_name, dataset, idx)
    

