#--------------------------------
# Import: Python libraries
#--------------------------------

import torch
import logging

#--------------------------------
# Initialize: Wavefront transform
#--------------------------------

class WavefrontTransform(object):
    def __init__(self, params):
        self.params = params.copy()
        logging.debug("custom_transforms.py - Initializing WavefrontTransform")

        # Set initialization strategy for the wavefront
        self.phase_initialization_strategy = params['phase_initialization_strategy']

        if self.phase_initialization_strategy == 0:
            logging.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = torch.ones(), Amplitude = Sample")
        else:
            logging.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = Sample, Amplitude = torch.ones()")

    def __call__(self,sample):
        c,w,h = sample.shape 
        if self.phase_initialization_strategy == 0:
            phases = torch.ones(c,w,h)
            amplitude = sample
        else:
            phases = sample
            amplitude = torch.ones(c,w,h)

        return amplitude * torch.exp(1j*phases)

class per_sample_normalize(object):                                                                    
    def __init__(self):                                                             
        logging.debug("custom_transforms.py - Initializing Normalize")
                                                                                            
    def __call__(self,sample):                                                              
        
        shape = sample.shape
        spatial_shape = shape[-2:]
        channel_shape = shape[:-2]
        sample = sample.view(channel_shape.numel(), *spatial_shape)
       
        
        min_vals = sample.min(dim=1)[0].min(dim=1)[0]
        min_vals = min_vals.view(min_vals.shape[0], 1,1)
        min_vals = min_vals.repeat(1,*spatial_shape) 

        sample = sample - min_vals

        max_vals = sample.max(dim=1)[0].max(dim=1)[0]
        max_vals = max_vals.view(max_vals.shape[0], 1,1)
        max_vals = max_vals.repeat(1,*spatial_shape)
    
        sample = sample / max_vals
        
        sample = sample.view(*channel_shape, *spatial_shape)

        return sample

#--------------------------------
# Initialize: Threshold transform
#--------------------------------

class Threshold(object):
    def __init__(self, threshold):
        logging.debug("custom_transforms.py - Initializing Threshold")
        self.threshold = threshold
        logging.debug("custom_transforms.py | Threshold | Setting threshold to {}".format(self.threshold))

    def __call__(self, sample):
        return (sample > self.threshold)
