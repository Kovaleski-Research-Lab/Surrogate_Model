import pickle
import numpy
import sys
import os
import numpy as np

#### This file just confirms that the pillars are where we think they are.
#--- We convert

data_path = os.listdir("../results")
file_name = "000000.pkl"

file_path = os.path.join("../results", file_name)
        
with open(file_path, 'rb') as f:
    data = pickle.load(f)

## These values are hardcoded, taken from params
cell_min = 0
cell_max = 4.92

## These values are also hardcorded, but can be obtained dynamically
pix_min = 0   
pix_max = 395  # data['eps_data'].squeeze().shape[2]

## meep's geometries are defined with (0,0,0) at the origin. These values are obtained by placing the geometries on a plane with z=0 at the bottom
pillar_top = 2.58  #um 
pillar_bottom = 1.56  #um

to_pix = [pillar_top, pillar_bottom]

def convert(value, min_value, max_value, new_min, new_max):
    # Calculate the normalized value using linear interpolation
    converted_value = ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
    return converted_value

converted_values = [convert(value, cell_min, cell_max, pix_min, pix_max) for value in to_convert]

for original_value, converted_value in zip(to_convert, converted_values):
    print(f"Original Value in microns: {original_value}, Converted Value in pixels: {converted_value}")
