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

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import custom_transforms as ct
from core import preprocess_data

#--------------------------------
# Initialize: MNIST Wavefront
#--------------------------------
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
        train_file = 'pp_dataset.pt'
        valid_file = None
        test_file = None
        if stage == "fit" or stage is None:
            dataset = customDataset(self.params, torch.load(os.path.join(self.path_data, train_file)),
                                    self.transform)

            train_set_size = int(len(dataset)*0.8)
            valid_set_size = len(dataset) - train_set_size
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
        if params['source_wl'] == 1550:
            temp_near_fields = self.all_near_fields['near_fields_1550']
        elif params['source_wl'] == 1060:
            temp_near_fields = self.all_near_fields['near_fields_1060']
        elif params['source_wl'] == 1300:
            temp_near_fields = self.all_near_fields['near_fields_1300']
        elif params['source_wl'] == 1650:
            temp_near_fields = self.all_near_fields['near_fields_1650']
        elif params['source_wl'] == 2881:
            temp_near_fields = self.all_near_fields['near_fields_2881']
        temp_near_fields = torch.stack(temp_near_fields, dim=0)
        self.near_fields = temp_near_fields.squeeze(1)

        self.radii = data['radii']
        self.phases = data['phases']
        self.derivatives = data['derivatives']

        self.temp_flux_info = data['flux_info']
        self.flux_info = {'source_flux' : [], 'downstream_flux' : []}

        for item in self.temp_flux_info:
            self.flux_info['source_flux'].append(item['source_flux'])   # these values are ~ [0.14, 0.3]
            self.flux_info['downstream_flux'].append(item['downstream_flux']) # these values are in the same range as source_flux
        #self.transform = ct.per_sample_normalize()
        self.transform = None

    def __len__(self):
        return len(self.near_fields)

    def __getitem__(self, idx):
        if self.transform:   
            #return self.transform(self.near_fields[idx]), self.transform(self.far_fields[idx]), self.radii[idx].float(), self.phases[idx].float(), self.derivatives[idx].float()
            return (self.transform(self.near_fields[idx]), self.radii[idx].float(),
                        self.phases[idx].float(), self.derivatives[idx].float())
                        #self.flux_info['source_flux'][idx].float())
        else:   
            #return self.near_fields[idx], self.far_fields[idx], self.radii[idx].float(), self.phases[idx].float(), self.derivatives[idx].float()
            return (self.near_fields[idx], self.radii[idx].float(), self.phases[idx].float(),
                     self.derivatives[idx].float())
                    # self.flux_info['source_flux'][idx].float())

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
    dm.setup(pm, stage="fit")

    #View some of the data

    nf, radii, phase, derivative, source_flux = next(iter(dm.train_dataloader()))
    from IPython import embed; embed()

    #fig,ax = plt.subplots(1,2,figsize=(5,5))
    #ax[0].imshow(nf[2][1].abs().squeeze())
    #ax[1].imshow(ff[2][1].abs().squeeze())
    #plt.show()

