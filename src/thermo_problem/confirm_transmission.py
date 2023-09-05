import os
import pickle
import shutil 

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from IPython import embed

def get_fr_slice():

    # put value on a (0 to pm.cell_z) scale - meep defines the cell on a (-cell_z/2 to cell_z/2) scale
    value = fr_center
    value = value + cell_z / 2

    # length of the cell in microns
    cell_min = 0  # um
    cell_max = cell_z  # um

    # length of the cell in pixels
    pix_min = 0
    pix_max = eps_data.squeeze().shape[2]

    #pix_max = data['near_fields_1550']['ex'].squeeze().shape[2]
    temp = int(((value - cell_min) / (cell_max - cell_min)) * (pix_max - pix_min) + pix_min)

    pml_pix = (eps_data.squeeze().shape[2] - dfts[0][0].squeeze().shape[2]) // 2 # did i do this right?????
    #pml_pix = (data['eps_data'].squeeze().shape[2] - data['near_fields_1550']['ex'].squeeze().shape[2]) // 2
    return temp - pml_pix

def get_intensity(Ex, Ey, Ez):
    print(Ex.shape)
    E_0 = np.sqrt((abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2))
    I = 0.5 * E_0**2
    return(np.mean(I))

resolution = 80

n_fusedSilica = 1.44
n_PDMS = 1.4
n_amorphousSi = 3.48

a = 0.680   # lattice period 

pml_thickness = 0.780
height_pillar = 1.020
width_PDMS = 1.560
width_fusedSilica = 0.780

cell_x = a * 3
cell_y = a * 3
cell_z = round(2*pml_thickness + width_PDMS + height_pillar +  width_fusedSilica, 3)

center_PDMS = round(0.5*(height_pillar + width_PDMS + pml_thickness) 
                    + (pml_thickness + width_fusedSilica) - 0.5*cell_z, 3)
center_fusedSilica = round(0.5*(pml_thickness + width_fusedSilica) - 0.5*cell_z, 3)
center_pillar = round(pml_thickness + width_fusedSilica + 0.5*height_pillar
                    - 0.5*cell_z, 3) 

z_fusedSilica = pml_thickness + width_fusedSilica
z_PDMS = height_pillar + width_PDMS + pml_thickness

geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf,z_fusedSilica), 
                    center=mp.Vector3(0,0,center_fusedSilica),
                    material=mp.Medium(index=n_fusedSilica)),
            mp.Block(size=mp.Vector3(mp.inf,mp.inf,z_PDMS),
                    center=mp.Vector3(0,0,center_PDMS),
                    material=mp.Medium(index=n_PDMS))]

k_point = mp.Vector3(0,0,0)
#k_point = mp.Vector3(np.radians(35), 0, 0)  # k vector defines normal to the plane of propagation

wavelength = 1.55
freq = 1 / wavelength

## initialize matrix for data collection ##
##########################################

num = 9 # this is the number of pillars we will build
data = np.zeros((2,num))
dfts = []

## set up and build source ##
############################

center_source = round(pml_thickness + width_fusedSilica*0.2 - 0.5*cell_z, 3) 
source_cmpt = mp.Ey

fcen = freq
fwidth=0.7741935483870968

sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                            component = source_cmpt,
                            center=mp.Vector3(0,0,center_source),
                            size=mp.Vector3(cell_x,cell_y,0))]

fr_center = round(0.5*cell_z - pml_thickness - 0.3*width_PDMS, 3)

def run_sim():
    sim.run(until_after_sources = mp.stop_when_fields_decayed(50, source_cmpt, mp.Vector3(0, 0, fr_center), 1e-4))

cell_size = mp.Vector3(cell_x,cell_y,cell_z)
pml_layers = [mp.PML(thickness = pml_thickness, direction = mp.Z)]

sim = mp.Simulation(cell_size=cell_size,
                    geometry=geometry,
                    sources=sources,
                    k_point=k_point,
                    boundary_layers=pml_layers,
                    resolution=resolution)

df=0;nfreq=1

non_pml = 3.36
near_vol = mp.Volume(center = mp.Vector3(0,0,0),
                                size = mp.Vector3(cell_x, cell_y, non_pml))
cs = [mp.Ex, mp.Ey, mp.Ez]
freq_list = [0.346926354230067,0.6060606060606061,0.6451612903225806,0.7692307692307692,0.9433962264150942]

dft_object = sim.add_dft_fields(cs, freq_list, where=near_vol)

run_sim()

initial_dft = [sim.get_dft_array(dft_object, mp.Ex, 2),
               sim.get_dft_array(dft_object, mp.Ey, 2),
               sim.get_dft_array(dft_object, mp.Ez, 2)]
sim.reset_meep()

x_list = [-a, 0, a, -a, 0, a, -a, 0, a]
y_list = [a, a, a, 0, 0, 0, -a, -a, -a] 
radii_list = [0.07712845, 0.18148884, 0.16429774, 0.14465851, 0.13001893, 0.12072568, 0.09872153, 0.10021467, 0.230]

for i, neighbor in enumerate(radii_list):
    radius = neighbor
    x_dim = x_list[i]
    y_dim = y_list[i]
    geometry.append(mp.Cylinder(radius=radii_list[i],
                                height=height_pillar,
                                axis=mp.Vector3(0,0,1),
                                center=mp.Vector3(x_list[i], y_list[i], center_pillar),
                                material=mp.Medium(index=n_amorphousSi)))

pbar = tqdm(total=num,leave=False)
central_index = None
for i,radius in enumerate(np.linspace(0.075,0.25,num=num)):
    for j, pillar in enumerate(geometry):
        if pillar.center == mp.Vector3(0,0,center_pillar):
            central_index = j
            print(f"central_index = {j}")
    if central_index is not None:
        geometry.pop(central_index)
    print(f"central pillar: {radius}")
    geometry.append(mp.Cylinder(radius=radius,
                        height=height_pillar,
                        axis=mp.Vector3(0,0,1),
                        center=mp.Vector3(0,0,center_pillar),
                        material=mp.Medium(index=n_amorphousSi)))

    print(f"at iteration {i} the central pillar is at {geometry[central_index].center} with {geometry[central_index].radius}")
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        k_point=k_point,
                        boundary_layers=pml_layers,
                        #symmetries=symmetries,
                        resolution=resolution)
    
    dft_object = sim.add_dft_fields(cs, freq_list, where=near_vol)
    
    run_sim()
    
    data[0,i] = radius
    #data[1,i] = dfts 
    dfts.append([sim.get_dft_array(dft_object, mp.Ex, 2), sim.get_dft_array(dft_object, mp.Ey, 2), sim.get_dft_array(dft_object, mp.Ez, 2)])

    pbar.update(1)
pbar.close()

eps_data = sim.get_epsilon()

fr_slice = get_fr_slice()

initial_x = initial_dft[0]
initial_y = initial_dft[1]
initial_z = initial_dft[2]

initial_x = initial_x[:,:,fr_slice]
initial_y = initial_y[:,:,fr_slice]
initial_z = initial_z[:,:,fr_slice]

initial_I = get_intensity(initial_x, initial_y, initial_z)
print(f"initial intensity: {initial_I}")

dft_slices = []
for result in dfts:
    Ex = result[0]
    Ey = result[1]
    Ez = result[2]

    Ex = Ex[:,:,fr_slice]
    Ey = Ey[:,:,fr_slice]
    Ez = Ez[:,:,fr_slice]

    slice = [Ex, Ey, Ez]
    dft_slices.append(slice)

I_list = []
for slice in dft_slices:
    print(slice[0].shape, slice[1].shape, slice[2].shape)
    I = get_intensity(slice[0], slice[1], slice[2])
    print(round(I, 3), round((I/initial_I),3))
    I_list.append(I / initial_I)

radii = data[0,:]

#plt.style.use('seaborn')

results = {"radii": radii, "trans": I_list}
embed()
