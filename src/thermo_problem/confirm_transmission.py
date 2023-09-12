import yaml
import time
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
import os

sys.path.append("../3x3_pillar_sims/")
import _3x3Pillars
sys.path.append("../")
from utils import parameter_manager
from core import preprocess_data as pp

def get_intensity(Ex, Ey, Ez):
    #print(Ex.shape)
    E_0 = np.sqrt((abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2))
    I = 0.5 * E_0**2
    return(np.mean(I))

def get_initial_xyz(initial_dft, mon_slice):
    initial_x = initial_dft[0]
    initial_y = initial_dft[1]
    initial_z = initial_dft[2]
    
    initial_x = initial_x[:,:,mon_slice]
    initial_y = initial_y[:,:,mon_slice]
    initial_z = initial_z[:,:,mon_slice]
    return initial_x, initial_y, initial_z

if __name__=='__main__':
    params = yaml.load(open('../../src/config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.ParameterManager(params=params)
    print("success")

    a = pm.lattice_size
    model = _3x3Pillars._3x3PillarSim()
    model.build_geometry(pm.geometry_params)    
    
    pm.geometry = [model.fusedSilica_block, model.PDMS_block]

    x_list = [-a, 0, a, -a, 0, a, -a, 0, a]
    y_list = [a, a, a, 0, 0, 0, -a, -a, -a]
#    radii_list = [0.07712845, 0.18148884, 0.16429774, 0.14465851, 0.13001893, 0.12072568, 0.09872153, 0.10021467, 0.230]
    
    num = 3 # this is the number of pillars we will build
    data = np.zeros((2,num))

    ## initialize lists for raw dft fields. each index holds x, y, and z components from a 
    ## single wavelength.
    dfts_2881, dfts_1650, dfts_1550, dfts_1300, dfts_1060 = [], [], [], [], []
    
    ## initialize list for calculated intensities. should end up with len(I_{wl})=num
    I_final2881, I_final1650, I_final1550, I_final1300, I_final1060 = [], [], [], [], []
 
    ## initialize list for transmissions
    trans2881, trans1650, trans1550, trans1300, trans1060 = [], [], [], [], []

    model.build_source(pm.source_params)
    pm.source = model.source

    model.build_sim(pm.sim_params)

    model.build_dft_mon(pm.dft_params)
    
    model.run_sim(pm.sim_params)

    initial_dft_2881 = [model.sim.get_dft_array(model.monitor, mp.Ex, 0),
                   model.sim.get_dft_array(model.monitor, mp.Ey, 0),
                   model.sim.get_dft_array(model.monitor, mp.Ez, 0)]
    
    initial_dft_1650 = [model.sim.get_dft_array(model.monitor, mp.Ex, 1),
                   model.sim.get_dft_array(model.monitor, mp.Ey, 1),
                   model.sim.get_dft_array(model.monitor, mp.Ez, 1)]
    
    initial_dft_1550 = [model.sim.get_dft_array(model.monitor, mp.Ex, 2),
                   model.sim.get_dft_array(model.monitor, mp.Ey, 2),
                   model.sim.get_dft_array(model.monitor, mp.Ez, 2)]
    
    initial_dft_1300 = [model.sim.get_dft_array(model.monitor, mp.Ex, 3),
                   model.sim.get_dft_array(model.monitor, mp.Ey, 3),
                   model.sim.get_dft_array(model.monitor, mp.Ez, 3)]
    
    initial_dft_1060 = [model.sim.get_dft_array(model.monitor, mp.Ex, 4),
                   model.sim.get_dft_array(model.monitor, mp.Ey, 4),
                   model.sim.get_dft_array(model.monitor, mp.Ez, 4)]
    
    initial_dfts = [initial_dft_2881, initial_dft_1650, initial_dft_1550, initial_dft_1300, initial_dft_1060]
    initial_dfts = np.stack(initial_dfts, axis=0)

    # we just ran a simulation without pillars and collected initial dfts for each frequency. 
    model.sim.reset_meep()
    def populate_neighborhood(radii_list):    
        for i, neighbor in enumerate(radii_list):
            pm.radius = neighbor
            pm.x_dim = x_list[i]
            pm.y_dim = y_list[i]
            model.build_geometry(pm.geometry_params)
            pm.geometry.append(model.pillar)

    pbar = tqdm(total=num,leave=False)
    #central_index = None
    central_pillars = []
    temp = {}
    count = 0
    for i, unit in enumerate(pm.geometry):
        if hasattr(unit, 'radius'):
            count += 1
    print(f"there are {count} pillars")
    for i,radius in enumerate(np.linspace(0.075,0.25,num=num)):
        print(f"i = {i}, radius = {radius}")
        #for j, unit in enumerate(pm.geometry):  # loop through 
            #if hasattr(unit, 'radius'):
            #    print(f"found a pillar with {unit.radius}")
            #    if unit.center == mp.Vector3(0,0,pm.center_pillar):
            #        print(f"center pillar is {unit.radius} at {unit.center}")
            #        pm.geometry.pop(j)
            #        pm.x_dim = pm.y_dim = 0
            #        pm.radius = radius
            #        model.build_geometry(pm.geometry_params)
            #        pm.geometry.insert(j, model.pillar)

            #        print("popped geometry. Now:")
            #        for j, unit in enumerate(pm.geometry):
            #            if hasattr(unit, 'radius'):
            #                print(f"found a pillar with {unit.radius} at {unit.center}")
        radii_list = [radius, radius, radius, radius, radius, radius, radius, radius, radius]
        populate_neighborhood(radii_list)
        #embed();exit()

        model.build_sim(pm.sim_params)
        model.build_dft_mon(pm.dft_params)
        
        model.run_sim(pm.sim_params)
        
        data[0,i] = radius
        #data[1,i] = dfts 
        model.reset_field_info()
        model.collect_field_info()
        dfts_2881.append(np.stack([model.dft_field_ex_2881, model.dft_field_ey_2881, model.dft_field_ez_2881], axis=0))
        dfts_1650.append(np.stack([model.dft_field_ex_1650, model.dft_field_ey_1650, model.dft_field_ez_1650], axis=0))
        dfts_1550.append(np.stack([model.dft_field_ex_1550, model.dft_field_ey_1550, model.dft_field_ez_1550], axis=0))
        dfts_1300.append(np.stack([model.dft_field_ex_1300, model.dft_field_ey_1300, model.dft_field_ez_1300], axis=0))
        dfts_1060.append(np.stack([model.dft_field_ex_1060, model.dft_field_ey_1060, model.dft_field_ez_1060], axis=0))


        if(i==0): # we need to calculate and collect initial intensity

            eps_data = model.eps_data
            nf = model.dft_field_ex_1550
            
            mon_slice = pp.get_mon_slice(pm, eps_data=eps_data, nf=nf) # need to give this eps_data (after populating pillars) and nf should be an x, y, or z component of raw dft fields. 

            x0_2881,y0_2881,z0_2881 = get_initial_xyz(initial_dft_2881, mon_slice)
            x0_1650,y0_1650,z0_1650 = get_initial_xyz(initial_dft_1650, mon_slice)
            x0_1550,y0_1550,z0_1550 = get_initial_xyz(initial_dft_1550, mon_slice)
            x0_1300,y0_1300,z0_1300 = get_initial_xyz(initial_dft_1300, mon_slice)
            x0_1060,y0_1060,z0_1060 = get_initial_xyz(initial_dft_1060, mon_slice)
             
            initial_I_2881 = get_intensity(x0_2881, y0_2881, z0_2881)
            initial_I_1650 = get_intensity(x0_1650, y0_1650, z0_1650)
            initial_I_1550 = get_intensity(x0_1550, y0_1550, z0_1550)
            initial_I_1300 = get_intensity(x0_1300, y0_1300, z0_1300)
            initial_I_1060 = get_intensity(x0_1060, y0_1060, z0_1060)

            all_initialI = [initial_I_2881, initial_I_1650, initial_I_1550, initial_I_1300, initial_I_1060]
            temp['all_initialI'] = all_initialI
            temp['neighbors'] = radii_list

        # we're dumping as we go in case the program quits due to intensive memory usage
        dfts = [model.dft_field_ex_2881, model.dft_field_ey_2881, model.dft_field_ez_2881]
        x,y,z = get_initial_xyz(dfts, mon_slice)
        I_final2881.append(get_intensity(x,y,z))
        trans2881.append(get_intensity(x,y,z) / initial_I_2881)
        
        dfts = [model.dft_field_ex_1650, model.dft_field_ey_1650, model.dft_field_ez_1650]
        x,y,z = get_initial_xyz(dfts, mon_slice)
        I_final1650.append(get_intensity(x,y,z))
        trans1650.append(get_intensity(x,y,z) / initial_I_1650)

        dfts = [model.dft_field_ex_1550, model.dft_field_ey_1550, model.dft_field_ez_1550]
        x,y,z = get_initial_xyz(dfts, mon_slice)
        I_final1550.append(get_intensity(x,y,z))
        trans1550.append(get_intensity(x,y,z) / initial_I_1550)

        dfts = [model.dft_field_ex_1300, model.dft_field_ey_1300, model.dft_field_ez_1300]
        x,y,z = get_initial_xyz(dfts, mon_slice)
        I_final1300.append(get_intensity(x,y,z))
        trans1300.append(get_intensity(x,y,z) / initial_I_1550)

        dfts = [model.dft_field_ex_1060, model.dft_field_ey_1060, model.dft_field_ez_1060]
        x,y,z = get_initial_xyz(dfts, mon_slice)
        I_final1060.append(get_intensity(x,y,z))
        trans1060.append(get_intensity(x,y,z) / initial_I_1060)
        
        temp['I_2881'] = I_final2881
        temp['I_1650'] = I_final1650
        temp['I_1550'] = I_final1550
        temp['I_1300'] = I_final1300
        temp['I_1060'] = I_final1060
        temp['central_pillars'] = central_pillars
        
        temp['trans_2881'] = trans2881
        temp['trans_1650'] = trans1650
        temp['trans_1550'] = trans1550
        temp['trans_1300'] = trans1300
        temp['trans_1060'] = trans1060 
    
        timestamp = time.strftime("%M%S")
        #with open(f"temp/temp_results{timestamp}.pkl", "wb") as f:
        #    pickle.dump(temp, f)
    
        if radius != 0.25:
            model.sim.reset_meep()
        pbar.update(1)

    pbar.close()
   
    results = {
            'initial' : all_initialI,
            'final' : [I_final2881, I_final1650, I_final1550, I_final1300, I_final1060],
            'transmission' : [trans2881, trans1650, trans1550, trans1300, trans1060],
    }
    import pandas as pd
    df = pd.DataFrame(results)
    #df['difference'] = 
    df.index = [2881, 1650, 1550, 1300, 1060]

    print(df)
    
    with open(f"temp/final_results_LPA.pkl","wb") as f:
        pickle.dump(results, f) 
        print(f"initial intensities: 2881: {initial_I_2881}, 1650: {initial_I_1650}, 1550: {initial_I_1550}, 1300: {initial_I_1300}, 1060: {initial_I_1060}")
