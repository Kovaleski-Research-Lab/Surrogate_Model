import sys 
import logging
import time
import os
import shutil
from IPython import embed
import yaml
import meep as mp
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

import _3x3Pillars
sys.path.append("../")
from utils import parameter_manager

def create_folder(path):

    if(not(os.path.exists(path))):
        os.makedirs(path)

def dump_geometry_image(model, pm):
    plt.figure()
    plot_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(pm.cell_x, 0, pm.cell_z))    
    model.sim.plot2D(output_plane = plot_plane)
    plt.savefig("geometry.png")

def dump_data(neighbor_index, data, pm): # this is called when we're generating data
     
    folder_path_sims = pm.path_dataset

    sim_name = "%s.pkl" % (str(neighbor_index).zfill(6))
    filename_sim = os.path.join(folder_path_sims, sim_name)

    with open(filename_sim, "wb") as f:
        pickle.dump(data,f)
   
    # Make sure pickle is written  

    time.sleep(20)
    
    print("\nEverything done\n")

def run(radii_list, index, pm, dataset=None):
    #run(radii_list, idx, pm, resim, exp_name, dataset) # idx identifies the index of results we're getting a resim for
    a = pm.lattice_size
    
    # Initialize model #
    model = _3x3Pillars._3x3PillarSim()

    # Build geometry for initial conditions (no pillar) #
    model.build_geometry(pm.geometry_params)
    pm.geometry = [model.fusedSilica_block, model.PDMS_block]
   
    # should make this general, so it is dependent on grid size (currently hardcoded for 3x3) 
    x_list = [-a, 0, a, -a, 0, a, -a, 0, a]
    y_list = [a, a, a, 0, 0, 0, -a, -a, -a]
 
    for i, neighbor in enumerate(radii_list):
        pm.radius = neighbor
        pm.x_dim = x_list[i]
        pm.y_dim = y_list[i]
        model.build_geometry(pm.geometry_params)
        pm.geometry.append(model.pillar)

    # Build Source object #
    model.build_source(pm.source_params)
     
    # Build Simulation object # 
    pm.source = model.source
    model.build_sim(pm.sim_params)

    # Build DFT monitor and populate field info #
    model.build_dft_mon(pm.dft_params)  
    start_time = time.time()
    model.run_sim(pm.sim_params)
    elapsed_time = time.time() - start_time
    elapsed_time = round(elapsed_time / 60,2)

    model.collect_field_info()
       
    data = {}

    data["near_fields_1550"] = {}
    data["near_fields_1550"]["ex"] = model.dft_field_ex_1550
    data["near_fields_1550"]["ey"] = model.dft_field_ey_1550
    data["near_fields_1550"]["ez"] = model.dft_field_ez_1550
    
    data["near_fields_1060"] = {}
    data["near_fields_1060"]["ex"] = model.dft_field_ex_1060
    data["near_fields_1060"]["ey"] = model.dft_field_ey_1060
    data["near_fields_1060"]["ez"] = model.dft_field_ez_1060

    data["near_fields_1300"] = {}
    data["near_fields_1300"]["ex"] = model.dft_field_ex_1300
    data["near_fields_1300"]["ey"] = model.dft_field_ey_1300
    data["near_fields_1300"]["ez"] = model.dft_field_ez_1300

    data["near_fields_1650"] = {}
    data["near_fields_1650"]["ex"] = model.dft_field_ex_1650
    data["near_fields_1650"]["ey"] = model.dft_field_ey_1650
    data["near_fields_1650"]["ez"] = model.dft_field_ez_1650

    data["near_fields_2881"] = {}
    data["near_fields_2881"]["ex"] = model.dft_field_ex_2881
    data["near_fields_2881"]["ey"] = model.dft_field_ey_2881
    data["near_fields_2881"]["ez"] = model.dft_field_ez_2881

    data["eps_data"] = model.eps_data
    data["sim_time"] = elapsed_time
    data["radii"] = radii_list
    
    if(pm.resim == 0):
        dump_data(index, data, pm) 
    else:
        embed()
        eval_name = f"sample_{idx}.pkl"
        path_results = "/develop/results/spie_journal_2023"
        path_resim = os.path.join(path_results, pm.exp_name + "_2", dataset + "_info") 
        filename = os.path.join(path_resim, eval_name)
        embed();exit()
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    #dump_geometry_image(model, pm)
if __name__=="__main__":

    # Run experiment
 
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.ParameterManager(params=params)
    print(f"resolution is {pm.resolution}")

    parser = argparse.ArgumentParser()
    if pm.resim == 1:
        parser.add_argument("-r0", type=float)
        parser.add_argument("-r1", type=float)
        parser.add_argument("-r2", type=float)
        parser.add_argument("-r3", type=float)
        parser.add_argument("-r4", type=float)
        parser.add_argument("-r5", type=float)
        parser.add_argument("-r6", type=float)
        parser.add_argument("-r7", type=float)
        parser.add_argument("-r8", type=float)
        parser.add_argument("-index", type=int, help="Index - neighbor index if generating data, if doing resim, a value between 0 and 8") 
        parser.add_argument("-dataset", type=str, help="Train or valid - if we're doing resims")

        args = parser.parse_args()
        r0 = args.r0
        r1 = args.r1
        r2 = args.r2
        r3 = args.r3
        r4 = args.r4
        r5 = args.r5
        r6 = args.r6
        r7 = args.r7
        r8 = args.r8
        idx = args.index
        dataset = args.dataset

        radii_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8] 
        run(radii_list, idx, pm, dataset) # idx identifies the index of results we're getting a resim for

    elif(pm.resim == 0): # we are generating data.
        parser.add_argument("-index", type=int, help="The index matching the index in radii_neighbors")
        parser.add_argument("-path_out_sims", help="This is the path that simulations get dumped to") # this is empty in our config file. gets set in the kubernetes job file
       
        args = parser.parse_args() 
        params['path_dataset'] = args.path_out_sims
        idx = args.index 

        neighbors_library = pickle.load(open("neighbors_library_allrandom.pkl", "rb"))
        radii_list = neighbors_library[idx]
        run(radii_list, idx, pm, dataset)
         
    else:
        print("Set resim in config.yaml to 0 for data generation and 1 to run a resim")
        exit()    
