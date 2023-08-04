import os
import sys
import yaml
import torch 
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('../')
from utils import parameter_manager
from core import propagator

if __name__ == "__main__":

    params = yaml.load(open('../config.yaml', 'r'), Loader = yaml.FullLoader)
    distance = 10.e-9    
    params['distance'] = distance
    pm = parameter_manager.Parameter_Manager(params = params)
    prop = propagator.Propagator(pm.params_propagator)

    data = torch.load('../data/preprocessed/cai_data.pt')
    
    nf = data['near_fields']
    ff = prop(nf)
    max_val = torch.max(ff[0][1].abs())
    min_val = torch.min(ff[0][1].abs())
    for i in tqdm(range(1,1000)):
        ff = prop(nf, distance = torch.tensor(distance * i))
        fig,ax = plt.subplots(1,1,figsize=(5,5)) 
        im = ax.imshow(ff[0][1].squeeze().abs())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title("Distance = {:.3E}".format(pm.distance * i))
        plt.tight_layout()
        fig.savefig('images/{:05d}.png'.format(i))
        plt.close('all') 
        
