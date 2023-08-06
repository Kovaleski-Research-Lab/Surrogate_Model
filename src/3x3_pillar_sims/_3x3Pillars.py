import os
import pickle
import shutil 
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from IPython import embed
import yaml
import sys
import logging
import time

#sys.path.append("../")
from utils import parameter_manager

def mod_axes(ax):

    font = {'size': 14}

    ax.set_xlabel('X [$\mu$m]', fontdict = font)
    ax.set_ylabel('Z [$\mu$m]', fontdict = font)
    ax.tick_params(axis='both', labelsize = 14)
    return ax

class _3x3PillarSim():
    def __init__(self):

        logging.debug(" Initializing Single Pillar Sim")
        
    def build_geometry(self, params):
        
        self.fusedSilica_block = mp.Block(size=mp.Vector3(mp.inf,mp.inf,params['z_fusedSilica']), 
                                    center=mp.Vector3(0,0,params['center_fusedSilica']),
                                    material=mp.Medium(index=params['n_fusedSilica']))
                    
        self.PDMS_block =        mp.Block(size=mp.Vector3(mp.inf,mp.inf,params['z_PDMS']),
                                    center=mp.Vector3(0,0,params['center_PDMS']),
                                    material=mp.Medium(index=params['n_PDMS']))
        if params['radius'] is not None:
            self.pillar =           mp.Cylinder(radius=params['radius'],
                                        height=params['height_pillar'],
                                        axis=mp.Vector3(0,0,1),
                                        center=mp.Vector3(params['x_dim'],params['y_dim'],params['center_pillar']),
                                        material=mp.Medium(index=params['n_amorphousSi'])) 

    def build_source(self, params):
        print("building source")
        if params['source_type'] == "gaussian":
            self.source = [mp.Source(mp.GaussianSource(params['fcen'],
                            fwidth=params['fwidth']),
                            component = params['source_cmpt'],
                            center=mp.Vector3(0,0,params['center_source']),
                            size=mp.Vector3(params['cell_x'],params['cell_y'],0))]

        if params['source_type'] == "continuous":
            self.source = [mp.Source(mp.ContinuousSource(frequency=params['freq']),
                            component=params['source_cmpt'],
                            center=mp.Vector3(0,0,params['center_source']),
                            size=mp.Vector3(params['cell_x'],params['cell_y'],0))]

    def build_sim(self, params):
        print("building sim")
        self.sim = mp.Simulation(cell_size=params['cell_size'],
                                geometry = params['geometry'],
                                sources = self.source,
                                k_point = params['k_point'],
                                boundary_layers = params['pml_layers'],
                                symmetries = params['symmetries'],
                                resolution = params['resolution'])

    def build_flux_mon(self, params):
        print("building flux monitor")
        self.downstream_flux_object = self.sim.add_flux(params['fcen'], 
                                             params['df'],
                                             params['nfreq'],
                                             params['fr'])

        # we want this flux monitor to be just past the source, not right on top of it.
        print("downstream_flux_object created")
        temp_fr_center = params['center_source']+ 0.1
        temp_fr = mp.FluxRegion(center = mp.Vector3(0,0,temp_fr_center),
                                            size = mp.Vector3(params['cell_x'],params['cell_y'], 0))

        self.source_flux_object = self.sim.add_flux(params['fcen'],
                                            params['df'],
                                            params['nfreq'],
                                            temp_fr)
        print("source_flux_object created")

    def build_dft_mon(self, params):

        cs = [mp.Ex, mp.Ey, mp.Ez]
        freq = params['freq_list']
        where = params['near_vol']
         
        self.near_fields = self.sim.add_dft_fields(cs, freq, where=where)

 
    def collect_field_info(self, params):
     
        # the third parameter (type=int) takes the index of params['freq_list']
        # params['freq_list'] is set in parameter_manager.calculate_dependencies()
        # and looks like: [freq_2881, freq_1650, freq_1550, freq_1300, freq_1060] 
        self.dft_field_ex_2881 = self.sim.get_dft_array(self.near_fields, mp.Ex, 0)
        self.dft_field_ey_2881 = self.sim.get_dft_array(self.near_fields, mp.Ey, 0)
        self.dft_field_ez_2881 = self.sim.get_dft_array(self.near_fields, mp.Ez, 0)

        self.dft_field_ex_1650 = self.sim.get_dft_array(self.near_fields, mp.Ex, 1)
        self.dft_field_ey_1650 = self.sim.get_dft_array(self.near_fields, mp.Ey, 1)
        self.dft_field_ez_1650 = self.sim.get_dft_array(self.near_fields, mp.Ez, 1)

        self.dft_field_ex_1550 = self.sim.get_dft_array(self.near_fields, mp.Ex, 2)
        self.dft_field_ey_1550 = self.sim.get_dft_array(self.near_fields, mp.Ey, 2)
        self.dft_field_ez_1550 = self.sim.get_dft_array(self.near_fields, mp.Ez, 2)

        self.dft_field_ex_1300 = self.sim.get_dft_array(self.near_fields, mp.Ex, 3)
        self.dft_field_ey_1300 = self.sim.get_dft_array(self.near_fields, mp.Ey, 3)
        self.dft_field_ez_1300 = self.sim.get_dft_array(self.near_fields, mp.Ez, 3)

        self.dft_field_ex_1060 = self.sim.get_dft_array(self.near_fields, mp.Ex, 4)
        self.dft_field_ey_1060 = self.sim.get_dft_array(self.near_fields, mp.Ey, 4)
        self.dft_field_ez_1060 = self.sim.get_dft_array(self.near_fields, mp.Ez, 4)

        self.eps_data = self.sim.get_epsilon()

    def run_sim(self, params):
        print("running sim")
        if params['source_type'] == "gaussian":
            self.sim.run(until_after_sources = mp.stop_when_fields_decayed(dt = 50,
                                                        c = params['source_cmpt'],
                                                        pt = mp.Vector3(0, 0, params['fr_center']),
                                                        decay_by = params['decay_rate'])) # see gaussian_vs_continous.ipynb for justification of 1e-3 for decay rate

        elif params['source_type'] == "continuous":
            self.sim.run(until=200)
        
    def get_animation(self, params):
        # this method does not work with the current Dockerfile. We likely have a bug
        # related to ffmpeg.
 
        plot_plane = mp.Volume(center = mp.Vector3(0,0,0),
                                size = mp.Vector3(params['lattice_size'], 0, params['cell_z']))
        plot_modifiers = [mod_axes]

        f = plt.figure(dpi=100, figsize=(8,15))
        Animate = mp.Animate2D(output_plane = params['plot_plane'],
                                fields = params['source_cmpt'],
                                f = f,
                                realtime = False,
                                normalize = True,
                                plot_modifiers = plot_modifiers)
        if params['source_type'] == "gaussian":
            until_after_sources =  mp.stop_when_fields_decayed(dt=50,
                                                        c=params['source_cmpt'],
                                                        pt=mp.Vector3(0, 0, params['fr_center']),
                                                        decay_by=params['decay_rate'])

            self.sim.run(mp.at_every(0.1, Animate), until=50)    
        
        results_path = "/develop/results/spie_journal_2023"
        name = f"{params['source_type']}.mp4"
        filename = os.path.join(results_path, name)
        Animate.to_mp4(params['fps'], filename)
            
        
if __name__=="__main__":
    
    params = yaml.load(open('config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.Parameter_Manager(params=params)
      
    #pm.radius = 0
    #model = SinglePillarSim()
    #
    ## Build geometry for initial conditions (no pillar) #
    #model.build_geometry(pm.material_params)
    #pm.geometry = [model.fusedSilica_block, model.PDMS_block]
    #
    ## Build source #
    #pm.source_type = "gaussian"
    #model.build_source(pm.source_params)
    #
    ## Build initial simulation: Get initial flux #
    #pm.source = model.source
    #model.build_sim(pm.sim_params)

    ## Build flux monitor #    
    #model.build_flux_mon(pm.flux_params)
    #
    ## Run sim and get initial flux #
    #pm.decay_rate = 1e-4
    #model.run_sim(pm.source_params)
    #initial_flux = mp.get_fluxes(model.flux_object)[0]

    #model.sim.reset_meep()
    #    
    #num = 15
    #data = np.zeros((4, num))

    #pbar = tqdm(total=num, leave=False)
    #for i, rad in enumerate(np.linspace(params['rad_min'], params['rad_max'],num = num)):
    #    start_time = time.time()
    #    pm.radius = rad
    #    model.build_geometry(pm.material_params)
    #    pm.geometry.append(model.pillar)
    #    model.build_sim(pm.sim_params)
    #    model.build_flux_mon(pm.flux_params)
    #    model.run_sim(pm.source_params)
    #    elapsed_time = time.time() - start_time
    #    
    #    res = model.sim.get_eigenmode_coefficients(model.flux_object, [1], eig_parity=mp.ODD_Y)
    #    coeffs = res.alpha

    #    flux = abs(coeffs[0,0,0]**2)
    #    phase = np.angle(coeffs[0,0,0])

    #    data[0,i] = rad
    #    data[1,i] = flux
    #    data[2,i] = phase
    #    data[3,i] = elapsed_time

    #    if(rad != params['rad_max']):
    #        model.sim.reset_meep()
    #        print("rad = {}, i = {}".format(rad, i))
    #        pm.geometry.pop(-1)
    #    pbar.update(1)
    #pbar.close()
    #
    #data[1,0:] = data[1,0:] / initial_flux

    #path_data = "/develop/code/spie_journal_2023"
    #folder_name = "test_meep_sim"
    #name = "test"
    #
    #folder_path = os.path.join(path_data,folder_name)
    #file_name = os.path.join(folder_path, f"{name}.pkl")

    #if not os.path.exists(folder_path):
    #    os.makedirs(folder_path)
    #    print("Folder path created.")
    #else:
    #    print("Folder path already exists")

    #with open(file_name, 'wb') as file:
    #    pickle.dump(data, file)
   
