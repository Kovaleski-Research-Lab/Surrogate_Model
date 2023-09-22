#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import pickle
import os
import sys
import glob #TODO: Needed?
import torch
import logging
import numpy as np #TODO: Needed?
from typing import Optional
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from IPython import embed

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import custom_transforms as ct

def get_intensities(nf):
    temp = []
    for field in nf:
        mag_x = field[0,0,:,:]
        mag_y = field[1,0,:,:]
        mag_z = field[2,0,:,:]
        E_0 = torch.sqrt(torch.abs(mag_x)**2 + torch.abs(mag_y)**2 + torch.abs(mag_z)**2)
        intensity = 0.5 * E_0**2
        temp.append(torch.mean(intensity))
    return temp

class CAI_Datamodule(LightningDataModule):
    def __init__(self, params, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing CAI_DataModule")

        self.params = params.copy()
        self.n_cpus = self.params['n_cpus']

        self.path_data = self.params['path_data']
        self.path_root = self.params['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logging.debug("datamodule.py - Setting path_data to {}".format(self.path_data))
        self.batch_size = self.params['batch_size']
       
        self.transform = transform #TODO

        self.initialize_cpus(self.n_cpus)

    def initialize_cpus(self, n_cpus):
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count():
           n_cpus = 1
        self.n_cpus = n_cpus 
        logging.debug("CAI_DataModule | Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self):
        pass
        #preprocess_data.preprocess_data(path = os.path.join(self.path_data, 'raw'))

    def load_data(self, pkl_directory):
        nf = []
        radii = []
        phases = []
        derivatives = []
        #sim_times = []
        for pkl_file in os.listdir(pkl_directory):
            if pkl_file.endswith('.pkl'):
                # Load the data from the .pkl file
                with open(os.path.join(pkl_directory, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                    nf.append(data['all_near_fields'])
                    radii.append(data['radii'])
                    phases.append(data['phases'])
                    derivatives.append(data['derivatives'])
       #             sim_times.append(data['sim_times'])
        
        data = {'all_near_fields' : nf,    
                'radii' : radii, 
                'phases' : phases,
                'derivatives' : derivatives,
                #'sim_times' : sim_times,
                }
        
        return data

    def setup(self, stage: Optional[str] = None):
        #TODO

        # this next block is a bandaid. the model expects a .pt file but preprocess.py was changed
        # to dump out .pkl files to work better with kubernetes.
        # ---- Need to overhaul the data preprocess step for this model! ----
        pkl_directory = '/develop/data/spie_journal_2023/kube_dataset/preprocessed'

        new_data = {
                    'all_near_fields': {
                                    'near_fields_1550': None, 
                                    'near_fields_1060': None,
                                    'near_fields_1300': None,
                                    'near_fields_1650': None,
                                    'near_fields_2881': None,
                                    },
                    'radii': [],
                    'phases': [],
                    'derivatives': [],
                    #'sim_times': [],
                    }

        pkl_data = self.load_data(pkl_directory)
        
        temp_1550, temp_1060, temp_1300, temp_1650, temp_2881 = [], [], [], [], []
        for element in pkl_data['all_near_fields']:  # looping through a list
            
            # each element is a dictionary
            for key, value in element.items():
                if key == 'near_fields_1550':
                    temp_1550.append(value)
                if key == 'near_fields_1650':
                    temp_1650.append(value)
                if key == 'near_fields_1300':
                    temp_1300.append(value)
                if key == 'near_fields_2881':
                    temp_2881.append(value)
                if key == 'near_fields_1060':
                    temp_1060.append(value)

        new_data['all_near_fields']['near_fields_1550'] = temp_1550
        new_data['all_near_fields']['near_fields_1650'] = temp_1650
        new_data['all_near_fields']['near_fields_1300'] = temp_1300
        new_data['all_near_fields']['near_fields_1060'] = temp_1060
        new_data['all_near_fields']['near_fields_2881'] = temp_2881
        
        for radius in pkl_data['radii']:    
            new_data['radii'].append(radius.squeeze())
        for phase in pkl_data['phases']:    
            new_data['phases'].append(phase.squeeze())
        for der in pkl_data['derivatives']:
            new_data['derivatives'].append(der)
        #for time in pkl_data['sim_times']:
        #    new_data['sim_times'].append(time)
        # Specify the path and filename for the output .pt file
        filename = "testing.pt"
        output_pt_file = f'/develop/data/spie_journal_2023/kube_dataset/preprocessed/{filename}'
        
        # Save the combined data to a single .pt file
        torch.save(new_data, output_pt_file)

        train_file = filename
        #train_file = 'dataset.pt'
        valid_file = None
        test_file = None
        if stage == "fit" or stage is None:
            #dataset = customDataset(self.params, torch.load(os.path.join(self.path_data, train_file)),
            #                        self.transform)
            
            dataset = customDataset(self.params, torch.load(os.path.join(output_pt_file)),
                                    self.transform)
            train_set_size = int(len(dataset)*0.8)
            valid_set_size = len(dataset) - train_set_size

            # set minimum of batch size to 2 to prevent torch.squeeze() errors in objective function           
            while train_set_size % self.batch_size < 2 and valid_set_size % self.batch_size < 2:
                self.batch_size += 1
                if self.batch_size > 100:
                    exit()

            self.cai_train, self.cai_valid = random_split(dataset,
                                                    [train_set_size, valid_set_size])
        if stage == "test" or stage is None:
            #self.cai_test = customDataset(torch.load(os.path.join(self.path_data, test_file)), self.transform)
            self.cai_test = self.cai_valid

    def train_dataloader(self):
        return DataLoader(self.cai_train, batch_size=self.batch_size, num_workers=self.n_cpus,
                                                        persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cai_valid, batch_size=self.batch_size, num_workers=self.n_cpus,
                                                        persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.cai_test, batch_size=self.batch_size, num_workers=self.n_cpus)

#--------------------------------
# Initialize: CAI dataset
#--------------------------------
class customDataset(Dataset):
    def __init__(self, params, data, transform):
        logging.debug("datamodule.py - Initializing customDataset")
        self.transform = transform
        logging.debug("customDataset | Setting transform to {}".format(self.transform))
        self.all_near_fields = data['all_near_fields']
        #if params['source_wl'] == 1550:
        #    temp_near_fields = self.all_near_fields['near_fields_1550']
        #elif params['source_wl'] == 1060:
        #    temp_near_fields = self.all_near_fields['near_fields_1060']
        #elif params['source_wl'] == 1300:
        #    temp_near_fields = self.all_near_fields['near_fields_1300']
        #elif params['source_wl'] == 1650:
        #    temp_near_fields = self.all_near_fields['near_fields_1650']
        #elif params['source_wl'] == 2881:
        #    temp_near_fields = self.all_near_fields['near_fields_2881']
        
        temp_nf_2881 = self.all_near_fields['near_fields_2881']
        temp_nf_1650 = self.all_near_fields['near_fields_1650']
        temp_nf_1550 = self.all_near_fields['near_fields_1550']
        temp_nf_1300 = self.all_near_fields['near_fields_1300']
        temp_nf_1060 = self.all_near_fields['near_fields_1060']
        temp_nf_2881, temp_nf_1650, temp_nf_1550, temp_nf_1300, temp_nf_1060 = (
                                            torch.stack(temp_nf_2881, dim=0),
                                            torch.stack(temp_nf_1650, dim=0),
                                            torch.stack(temp_nf_1550, dim=0),
                                            torch.stack(temp_nf_1300, dim=0),
                                            torch.stack(temp_nf_1060, dim=0))

        self.nf_2881, self.nf_1650, self.nf_1550, self.nf_1300, self.nf_1060 = (
                                            temp_nf_2881.squeeze(1),
                                            temp_nf_1650.squeeze(1),
                                            temp_nf_1550.squeeze(1),
                                            temp_nf_1300.squeeze(1),
                                            temp_nf_1060.squeeze(1))
        # i have near fields with shape [num_samples, 3, 2, xdim, ydim]
        #                               x, y, z components, magnitude and angle 
        self.intensities_2881 = get_intensities(self.nf_2881)
        self.intensities_1650 = get_intensities(self.nf_1650)
        self.intensities_1550 = get_intensities(self.nf_1550)
        self.intensities_1300 = get_intensities(self.nf_1300)
        self.intensities_1060 = get_intensities(self.nf_1060)
     
        self.radii = data['radii']
        self.phases = data['phases']
        self.derivatives = data['derivatives']
        
        #self.transform = ct.per_sample_normalize()
        self.transform = None
    
    def __len__(self):
        return len(self.nf_1550)

    def __getitem__(self, idx):

        if self.transform:   
            pass
            #return (self.transform(self.nf_2881[idx], self.nf_1650[idx], self.nf_1550[idx],
            #            self.nf_1650[idx], self.nf_1300[idx], self.nf_1060[idx],
            #            self.radii[idx].float(),self.phases[idx].float(),
            #            self.derivatives[idx].float(), self.intensities[idx].float()))
        else:
            batch = {
                    'nf_2881'             : self.nf_2881[idx],
                    'nf_1650'             : self.nf_1650[idx],
                    'nf_1550'             : self.nf_1550[idx],
                    'nf_1300'             : self.nf_1300[idx],
                    'nf_1060'             : self.nf_1060[idx],
                    'radii'               : self.radii[idx].float(),
                    'phases'              : self.phases[idx].float(),
                    'derivatives'         : self.derivatives[idx].float(),
                    'intensities_2881'    : self.intensities_2881[idx].float(),
                    'intensities_1650'    : self.intensities_1650[idx].float(),
                    'intensities_1550'    : self.intensities_1550[idx].float(),
                    'intensities_1300'    : self.intensities_1300[idx].float(),
                    'intensities_1060'    : self.intensities_1060[idx].float(),
                    }
            return batch

#            return (self.nf_2881[idx], self.nf_1650[idx], self.nf_1550[idx], self.nf_1300[idx],
#                        self.nf_1060[idx], self.radii[idx].float(), self.phases[idx].float(),
#                        self.derivatives[idx].float(), self.intensities[idx].float())

#--------------------------------
# Initialize: Select dataset
#--------------------------------

def select_data(params):
    if params['which'] == 'CAI':
        return CAI_Datamodule(params) 
    else:
        logging.error("datamodule.py | Dataset {} not implemented!".format(params['which']))
        exit()

#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    import yaml
    import torch
    import matplotlib.pyplot as plt
    from utils import parameter_manager
    from pytorch_lightning import seed_everything

    logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    os.environ['SLURM_JOB_ID'] = '0'
    #plt.style.use(['science'])

    #Load config file   
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    params['model_id'] = 0

    #Parameter manager
    pm = parameter_manager.ParameterManager(params=params)

    #Initialize the data module
    dm = select_data(pm.params_datamodule)
    dm.prepare_data()
    #dm.setup(pm, stage="fit")
    dm.setup(stage="fit")

    #View some of the data

    batch = next(iter(dm.train_dataloader()))
    from IPython import embed; embed()

    #fig,ax = plt.subplots(1,2,figsize=(5,5))
    #ax[0].imshow(nf[2][1].abs().squeeze())
    #ax[1].imshow(ff[2][1].abs().squeeze())
    #plt.show()

