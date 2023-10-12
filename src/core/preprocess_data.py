import os
import sys
import yaml
import torch 
import pickle
import numpy as np
from tqdm import tqdm
#from numpy.polynomial.polynomial import Polynomial
from scipy import interpolate
from scipy.interpolate import BSpline
from IPython import embed

sys.path.append('../')
from utils import parameter_manager
from core import curvature

def get_Bsplines():

    # these discrete  values were generated using a single pillar simulation with souce wavelength 1550 nm. 
    # the simulation was conducted using meep (FDTD) with PML in the z direction and Bloch periodic boundary conditions in x and y.
    # the results rely on the local phase approximation. 
    radii = [0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225, 0.2375, 0.25]
    phase_list = [-3.00185845, -2.89738421, -2.7389328, -2.54946247, -2.26906522, -1.89738599, -1.38868364, -0.78489682, -0.05167712, 0.63232107, 1.22268106, 1.6775137, 2.04169308, 2.34964137, 2.67187105]
    radii = np.asarray(radii)
    phase_list = np.asarray(phase_list)
    
    # we use a BSpline interpolating function to complete the mapping from radii <-> phase.
    to_phase = interpolate.splrep(radii, phase_list, s=0, k=3)
    to_radii = interpolate.splrep(phase_list, radii, s=0, k=3)
    
    return to_phase, to_radii

def radii_to_phase(radii):
    to_phase, _ = get_Bsplines() 
    phases = torch.from_numpy(interpolate.splev(radii, to_phase))
    return phases

def phase_to_radii(phases):
    _, to_radii = get_Bsplines()
    radii = torch.from_numpy(interpolate.splev(phases, to_radii))
    return radii
 
# convert a value from micron location to pixel location in the cell. hard coded to the monitor
# center
def get_mon_slice(pm, data=None, eps_data=None, nf=None): # eps_data and nf are not assigned if we are preprocessing in batches.

    # put value on a (0 to pm.cell_z) scale - meep defines the cell on a (-cell_z/2 to cell_z/2) scale
    value = pm.mon_center
    value = value + pm.cell_z / 2

    # length of the cell in microns 
    cell_min = 0  # um
    cell_max = pm.cell_z  # um

    # length of the cell in pixels
    pix_min = 0   
    if data is not None: # hint: data is assigned if we are preprocessing in batches.
        pix_max = data['eps_data'].squeeze().shape[2]
    else:
        pix_max = eps_data.squeeze().shape[2]

    #pix_max = data['near_fields_1550']['ex'].squeeze().shape[2]
    temp = int(((value - cell_min) / (cell_max - cell_min)) * (pix_max - pix_min) + pix_min)

    if data is not None:
        pml_pix = (data['eps_data'].squeeze().shape[2] - data['near_fields_1550']['ex'].squeeze().shape[2]) // 2
    else:
        pml_pix = (eps_data.squeeze().shape[2] - nf.squeeze().shape[2]) // 2

    return temp - pml_pix

def get_intensity(Ex, Ey, Ez):

    E_0 = np.sqrt((abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2))
    I = 0.5 * E_0**2
    I = torch.from_numpy(I)
    return(torch.mean(I))

#def get_transmission(Ex, Ey, Ez, pm):
#    print(Ex.shape)
#    E_0 = np.sqrt((abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2))
#    I = 0.5 * E_0**2
#    mean = np.mean(I)
#    return(mean / pm.i_0)

# we retired this function because it relies on a polynomial interpolation which is less accurate and far less reliable at the limit
# than BSplines.
#def radii_to_phase(kube, radii):
#    if kube is True:
#        mapper = pickle.load(open("/develop/code/src/core/radii_to_phase.pkl", 'rb')) #KUBE
#    else:
#        mapper = pickle.load(open("/develop/code/src/surrogate_model/core/radii_to_phase.pkl", 'rb')) # LOCAL MARGE
#    phases = torch.from_numpy(Polynomial(mapper)(radii.numpy()))
#    phases =  torch.clamp(phases, min=-torch.pi, max=torch.pi)
#    return phases

def reconstruct_field(near_fields_mag, near_fields_angle):

    complex_field = near_fields_mag * torch.exp(1j * near_fields_angle)
    complex_field = complex_field.squeeze(dim=2)
    components = torch.split(complex_field, 1, dim=1)
    x_comp, y_comp, z_comp = components
    x_comp, y_comp, z_comp = [tensor.squeeze(0).squeeze(0) for tensor in (x_comp, y_comp, z_comp)]
    return x_comp, y_comp, z_comp 

def preprocess(pm, data, filename=None, kube=False):
    # make a dictionary of lists: one list for each source wavelength.
    wavelengths = ['1550', '1060', '1300', '1650', '2881']
    all_near_fields = {f'near_fields_{wl}': [] for wl in wavelengths}

    # the rest of the info can be stored in individual lists.
    
    eps_data = []
    #sim_times = []
    #intensities = []
    radii = []
    phases = []
    der = []

    # collect epsilon data, sim time, radii
    eps_data = torch.from_numpy(np.asarray(data['eps_data']))
    sim_time = torch.from_numpy(np.asarray(data['sim_time']))
                                                                                                                      
    radii = torch.from_numpy(np.asarray(data['radii'])).unsqueeze(dim=0)
    # convert radii to phase and collect phase
    #temp_phases = torch.from_numpy(radii_to_phase(radii[-1])) i used this to look at an already preprocessed dataset
    temp_phases = torch.from_numpy(np.asarray(radii_to_phase(radii[-1])))
    phases = temp_phases
    
    # calculate and collect derivatives
    temp_der = curvature.get_der_train(temp_phases.view(1,3,3))
    der = temp_der
    
    # organize near fields info by wavelength, get magnitude and phase info.
    nf_dict = {
                'nf_1550': data['near_fields_1550'], # dft_fields = data['near_fields']
                'nf_1060': data['near_fields_1060'],
                'nf_1300': data['near_fields_1300'],
                'nf_1650': data['near_fields_1650'],
                'nf_2881': data['near_fields_2881'],
              } 
    mon_slice = get_mon_slice(pm, data)
    
    for key, nf in nf_dict.items(): 
        # nf['ex'], ['ey'], and ['ex'] have shape (x, y, z). Need to get the z slice
        # corresponding to z = mon_center.
        temp_nf_ex = nf['ex'][:,:,mon_slice]
        temp_nf_ey = nf['ey'][:,:,mon_slice]
        temp_nf_ez = nf['ez'][:,:,mon_slice] 
        nf_ex = torch.from_numpy(temp_nf_ex).unsqueeze(dim=0)
        nf_ey = torch.from_numpy(temp_nf_ey).unsqueeze(dim=0)
        nf_ez = torch.from_numpy(temp_nf_ez).unsqueeze(dim=0)    
        temp = torch.cat([nf_ex, nf_ey, nf_ez], dim=0).unsqueeze(dim=0) 
        near_fields_mag = temp.abs().unsqueeze(dim=2) # contains mag of x, y, and z
        near_fields_angle = temp.angle().unsqueeze(dim=2) # contains angle of x, y, and z
        wl = ''.join(filter(str.isdigit, key))
        all_near_fields[f'near_fields_{wl}'] = torch.cat((near_fields_mag, near_fields_angle), dim=2)
                                                                                                                      
        # note: intensities are set in datamodule.py
    data = {'all_near_fields' : all_near_fields,    
            'radii' : radii, 
            'phases' : phases,                                                                                                   
            'derivatives' : der,
            #'sim_times' : sim_times,
            }
    if kube is True:
        path_save = '/develop/results/preprocessed' #KUBE
        pkl_file_path = os.path.join(path_save, filename)
        with open(pkl_file_path, "wb") as f:
            pickle.dump(data, f)
            print(filename + " dumped to preprocess folder.")
                                                                                                                                  
    else: # if kube is false we are either testing locally (on marge) or we're preprocessing a resim.
        # case 1: testing locally
        if pm.resim == 0: 
            path_save = '/develop/data/spie_journal_2023/kube_dataset/preprocessed'
            pkl_file_path = os.path.join(path_save, filename)
            with open(pkl_file_path, "wb") as f:
                pickle.dump(data, f)
                print(filename + " dumped to preprocess folder.")
        # case 2: doing a resim
        elif pm.resim == 1:
            return data

def preprocess_data(pm, kube, raw_data_files = None, path = None):
 
    if (kube is True):
        params = yaml.load(open('/develop/code/src/config.yaml', 'r'), #KUBE
                                        Loader = yaml.FullLoader)
    else:
        params = yaml.load(open('/develop/code/surrogate_model/src/config.yaml', 'r'), # LOCAL MARGE
                                    Loader = yaml.FullLoader)
    
    if raw_data_files is None:
        raw_data_files = os.listdir(path)

    count = 0
    #for f in raw_data_files:
    for f in tqdm(raw_data_files, desc="Preprocessing data"):
        if kube is True:
            path = "/develop/results/" # KUBE
        else:
            path = "/develop/data/spie_journal_2023/kube_dataset" # LOCAL MARGE
        try:
            filepath = os.path.join(path, f)
            print(f"loading in {filepath}...")
            with open(filepath, "rb") as file:
                data = pickle.load(file)
            print(f"got it: {file}")

            count += 1
            print(f"count = {count}, filename = {f}")
            filename = f
            
            preprocess(pm, data, filename, kube=kube) 
            
        except FileNotFoundError:
            print("pickle error: file not found")
        except pickle.PickleError:
            print("pickle error: pickle file error")
        except Exception as e:
            print("Some other error: ", e)
    print(count + " files preprocessed successfully.")
    
if __name__=="__main__":
    print("executing preprocess_data.py")
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    pm = parameter_manager.ParameterManager(params=params)

    kube = True 
    if kube is True:
        folder = os.listdir('/develop/results') # KUBE
    else:
        folder = os.listdir('/develop/data/spie_journal_2023/kube_dataset') # LOCAL MARGE

    raw_data_files = []
    for filename in folder:
        if filename.endswith(".pkl"):
            raw_data_files.append(filename)
    
    print(f'\nNumber of files to process: {len(raw_data_files)}')
    print(" ")
    preprocess_data(pm, kube, raw_data_files = raw_data_files)
    print("\nPreprocess complete")
    
    ## --- used this block to look at already preprocessed data and compare lagrangian -- #
    ## --- to BSpline (converting radii to phase).                                     -- #
    #data = torch.load("/develop/data/spie_journal_2023/kube_dataset/pp_dataset.pt")
    #radii = data['radii'].numpy()

    #phases = radii_to_phase(radii)
    #data['phases'] = phases    
    #embed()
