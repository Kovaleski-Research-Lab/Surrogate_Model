import os
import sys
import yaml
import torch 
import pickle
import numpy as np
from tqdm import tqdm
from numpy.polynomial.polynomial import Polynomial
from IPython import embed

sys.path.append('../')
from utils import mapping, parameter_manager
from core import curvature

# convert a value from micron location to pixel location in the cell. hard coded to the flux region
# center
def get_fr_slice(data, pm):

    # put value on a (0 to pm.cell_z) scale - meep defines the cell on a (-cell_z/2 to cell_z/2) scale
    value = pm.fr_center
    value = value + pm.cell_z / 2

    # length of the cell in microns 
    cell_min = 0  # um
    cell_max = pm.cell_z  # um

    # length of the cell in pixels
    pix_min = 0   
    pix_max = data['eps_data'].squeeze().shape[2]

    #pix_max = data['near_fields_1550']['ex'].squeeze().shape[2]
    temp = int(((value - cell_min) / (cell_max - cell_min)) * (pix_max - pix_min) + pix_min)

    pml_pix = (data['eps_data'].squeeze().shape[2] - data['near_fields_1550']['ex'].squeeze().shape[2]) // 2

    return temp - pml_pix

def radii_to_phase(kube, radii):
    if kube is True:
        mapper = pickle.load(open("/develop/code/src/core/radii_to_phase.pkl", 'rb')) #KUBE
    else:
        mapper = pickle.load(open("/develop/code/src/surrogate_model/core/radii_to_phase.pkl", 'rb')) # LOCAL MARGE
    phases = torch.from_numpy(Polynomial(mapper)(radii.numpy()))
    phases =  torch.clamp(phases, min=-torch.pi, max=torch.pi)
    return phases

def preprocess_data(pm, kube, raw_data_files = None, path = None):
 
    if (kube is True):
        params = yaml.load(open('/develop/code/src/config.yaml', 'r'), #KUBE
                                        Loader = yaml.FullLoader)
    else:
        params = yaml.load(open('/develop/code/surrogate_model/src/config.yaml', 'r'), # LOCAL MARGE
                                    Loader = yaml.FullLoader)
    
    # make a dictionary of lists: one list for each source wavelength.
    wavelengths = ['1550', '1060', '1300', '1650', '2881']
    all_near_fields = {f'near_fields_{wl}': [] for wl in wavelengths}

    # the rest of the info can be stored in individual lists.
    
    eps_data = []
    sim_times = []
    flux_info = []
    radii = []
    phases = []
    der = []

    if raw_data_files is None:
        raw_data_files = os.listdir(path)

    count = 0
    for f in tqdm(raw_data_files, desc="Preprocessing data"):
        if kube is True:
            path = "/develop/results/" # KUBE
        else:
            path = "/develop/data/spie_journal_2023/testing_new_dataset" # LOCAL MARGE
        data = pickle.load(open(os.path.join(path, f), "rb"))
        #embed(); exit()
        count += 1
        print(f"count = {count} file = {f}")

        # collect epsilon data, sim time, flux, radii
        eps_data.append(torch.from_numpy(np.asarray(data['eps_data'])))
        sim_times.append(torch.from_numpy(np.asarray(data['sim_time'])))

        # converting flux data to np.float16 to save space
        flux = {}
        flux['source_flux'] = torch.from_numpy(np.asarray(data['flux']['source_flux']))
        flux['downstream_flux'] = torch.from_numpy(np.asarray(data['flux']['downstream_flux']))
        flux_info.append(flux)
        radii.append(torch.from_numpy(np.asarray(data['radii'])).unsqueeze(dim=0))
        
        # convert radii to phase and collect phase
        temp_phases = torch.from_numpy(mapping.radii_to_phase(radii[-1]))
        phases.append(temp_phases)
        
        # calculate and collect derivatives
        temp_der = curvature.get_der_train(temp_phases.view(1,3,3))
        der.append(temp_der)
        
        # organize near fields info by wavelength, get magnitude 
        # and phase info.
        nf_dict = {
                    'nf_1550': data['near_fields_1550'], # dft_fields = data['near_fields']
                    'nf_1060': data['near_fields_1060'],
                    'nf_1300': data['near_fields_1300'],
                    'nf_1650': data['near_fields_1650'],
                    'nf_2881': data['near_fields_2881'],
                  } 
        fr_slice = get_fr_slice(data, pm)
        embed();exit();
        for key, nf in nf_dict.items(): 
            # nf['ex'], ['ey'], and ['ex'] have shape (x, y, z). Need to get the z slice
            # corresponding to z = fr_center.
            nf_ex = nf['ex'][:,:,fr_slice]
            nf_ey = nf['ey'][:,:,fr_slice]
            nf_ez = nf['ez'][:,:,fr_slice]
            nf_ex = torch.from_numpy(nf_ex).unsqueeze(dim=0)
            nf_ey = torch.from_numpy(nf_ey).unsqueeze(dim=0)
            nf_ez = torch.from_numpy(nf_ez).unsqueeze(dim=0)    
            temp = torch.cat([nf_ex, nf_ey, nf_ez], dim=0).unsqueeze(dim=0)
            near_fields_mag = temp.abs().unsqueeze(dim=2)
            near_fields_angle = temp.angle().unsqueeze(dim=2)
            wl = ''.join(filter(str.isdigit, key))
            all_near_fields[f'near_fields_{wl}'].append(torch.cat((near_fields_mag, near_fields_angle), dim=2))

    for key, nf in all_near_fields.items():
        nf = torch.cat(nf, dim=0).float()
    eps_data = torch.cat(eps_data, dim=0).float()
    radii = torch.cat(radii, dim=0).float()
    phases = torch.cat(phases, dim=0).float()
    der = torch.stack(der).float()

    data = {'all_near_fields' : all_near_fields,    
            #'eps_data' : eps_data,
            'radii' : radii, 
            'phases' : phases,
            'derivatives' : der,
            'sim_times' : sim_times,
            'flux_info' : flux_info,
            }

    if kube is True:
        path_save = '/develop/results/preprocessed' #KUBE
    else:
        path_save = '/develop/data/spie_journal_2023/testing_new_dataset'
    torch.save(data, os.path.join(path_save, 'final_test.pt'))

if __name__=="__main__":
    print("about to load params")
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.ParameterManager(params=params)

    kube = True
    print("kube = True") 
    if kube is True:
        folder = os.listdir('/develop/results') # KUBE
    else:
        folder = os.listdir('/develop/data/spie_journal_2023/testing_new_dataset')

    raw_data_files = []
    for filename in folder:
        if filename.endswith(".pkl"):
            raw_data_files.append(filename)
    print(f'\nFiles to process: {raw_data_files}')
    print(" ")
    preprocess_data(pm, kube, raw_data_files = raw_data_files)
    print("\nPreprocess complete")
