#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
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

#nf_mag_x = freq[0,0,:,:]
#nf_mag_y = freq[1,0,:,:]
#nf_mag_z = freq[2,0,:,:]
#nf_angle_x = freq[0,1,:,:]
#nf_angle_y = freq[1,1,:,:]
#nf_angle_z = freq[2,1,:,:]
#complex_field_x = nf_mag_x * torch.exp(1j * nf_angle_x)
#complex_field_y = nf_mag_y * torch.exp(1j * nf_angle_y)
#complex_field_z = nf_mag_z * torch.exp(1j * nf_angle_z)
#E_0 = torch.sqrt((abs(complex_field_x)**2 + abs(complex_field_y)**2 + abs(complex_field_z)**2))
#I = 0.5 * E_0**2
                                                                                                 
# check:
#x_mag = freq[0,0,:,:]
#y_mag = freq[1,0,:,:]
#z_mag = freq[2,0,:,:]
#E_0 = torch.sqrt(x_mag**2 + y_mag**2 + z_mag**2)
#intensity = 0.5 * E_0**2
#intensity = intensity.mean()
#temp.append(intensity)

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



#def get_intensities(nf):
#    list = []
#    mag = nf[:,:,0,:,:]
#    angle = nf[:,:,1,:,:]
#    for m, a in zip(mag, angle):
#
#        complex_field = m * torch.exp(1j * a) # shape = 3,166,166
#        components = torch.split(complex_field, 1, dim=0)
#        x_comp, y_comp, z_comp = components
#        x_comp, y_comp, z_comp = [tensor.squeeze(0) for tensor in (x_comp, y_comp, z_comp)]
#        
#        E_0 = torch.sqrt((abs(x_comp)**2 + abs(y_comp)**2 + abs(z_comp)**2))
#        I = 0.5 * E_0**2
#        list.append(torch.mean(I))
#    return list


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

    def setup(self, stage: Optional[str] = None):
        #TODO
        train_file = 'dev.pt'
        valid_file = None
        test_file = None
        if stage == "fit" or stage is None:
            dataset = customDataset(self.params, torch.load(os.path.join(self.path_data, train_file)),
                                    self.transform)
            
            train_set_size = int(len(dataset)*0.8)
            valid_set_size = len(dataset) - train_set_size

            # set minimum of batch size to 2            
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

