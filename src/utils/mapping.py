import os
import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import lagrange
from scipy.interpolate import BSpline

# phase_to_radii() and radii_to_phase() return numpy arrays with float64 values. 
# output shape = (n,)

def get_mapping(which):

    radii = [0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225, 0.2375, 0.25]
    phase_list = [-3.00185845, -2.89738421, -2.7389328, -2.54946247, -2.26906522, -1.89738599, -1.38868364, -0.78489682, -0.05167712, 0.63232107, 1.22268106, 1.6775137, 2.04169308, 2.34964137, 2.67187105]

    radii = np.asarray(radii)
    phase_list = np.asarray(phase_list)

    if(which=="to_phase"):
        tck = interpolate.splrep(radii, phase_list, s=0, k=3)
    
    elif(which=="to_rad"):
        tck = interpolate.splrep(phase_list, radii, s=0, k=3)

    return tck 

def phase_to_radii(phase_list):
    
    mapper = get_mapping("to_rad")
    to_radii = []
    for phase in phase_list:
        to_radii.append(interpolate.splev(phase_list,mapper))

    return np.asarray(to_radii[0])   

def radii_to_phase(radii):
    
    mapper = get_mapping("to_phase")
    to_phase = []
    for radius in radii:    
        to_phase.append(interpolate.splev(radii,mapper))

    return np.asarray(to_phase[0])

if __name__=="__main__":
    ### test use
    min_rad=0.07
    max_rad=0.3
    
    min_phi = -3.14
    max_phi = 3.14
    #test_radii = np.random.uniform(min_rad, max_rad, size=50)
    #test_phase = np.random.uniform(min_phi, max_phi, size=50)
    
    test_radii = [0.26011755, 0.20069265, 0.22208114, 0.16709304, 0.1543911 ,
           0.28266339, 0.15418758, 0.12698411, 0.18630338, 0.07128922,
           0.08515119, 0.28276876, 0.1874754 , 0.19871777, 0.29364868,
           0.08885194, 0.11002082, 0.23279479, 0.19273338, 0.23394921,
           0.23359128, 0.11537965, 0.18942803, 0.22491449, 0.24051975,
           0.07980938, 0.20139536, 0.27121413, 0.26102504, 0.09438579,
           0.07234532, 0.28393447, 0.1788532 , 0.1308121 , 0.1405294 ,
           0.08156689, 0.13329983, 0.24070562, 0.13309768, 0.2379797 ,
           0.14189078, 0.18961825, 0.26578843, 0.13884416, 0.20599239,
           0.08805529, 0.16411424, 0.16033523, 0.20230833, 0.13681477]
    
    test_phase = [ 1.51799213, -0.68962107,  0.07319133,  0.48363524, -2.72281699,
           -0.83721823,  0.04110824,  1.9794331 ,  1.87018232, -2.88178634,
            1.00186858, -1.75020749,  2.36503301, -0.93884895, -0.95952454,
           -0.05647852, -2.31985803,  1.43686339, -2.63723002, -1.15453557,
           -0.66780749, -1.88484533, -1.00140753,  1.47572927, -2.06702869,
           -2.17758571,  1.62943488, -1.18803155, -2.05472809, -1.68716965,
            0.36799487,  2.67194764,  0.72302552,  2.38176686,  3.05357965,
           -2.5667818 , -2.09934952,  1.68448547, -3.13134313, -0.69116234,
           -1.5116532 ,  1.20520053,  2.09879213, -1.28011675,  2.71328443,
           -2.8312563 ,  2.5834879 , -1.7613506 , -1.94590795,  0.90309074]
    
    to_phase = radii_to_phase(test_radii)
    
    to_radii = phase_to_radii(test_phase)
    
    from IPython import embed
    embed()
 

