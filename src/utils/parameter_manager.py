import meep as mp
import logging
import yaml
import sys
import traceback
from IPython import embed

class ParameterManager():
    def __init__(self, config = None, params = None):

        logging.debug("parameter_manager.py - Initializing Parameter_Manager")
    
        if config is not None:
            self.open_config(config)

        if params is not None:
            self.params = params.copy()

        self.parse_params(self.params)
        self.calculate_dependencies()
        self.param_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        param_dict = self.collect_params()
        if self.param_index >= len(param_dict):
            raise StopIteration
        key = list(param_dict.keys())[self.param_index]
        value = param_dict[key]
        self.param_index += 1
        return key, value

    def open_config(self, config_file):

        try:
            with open(config_file) as c:
                self.params = yaml.load(c, Loader = yaml.FullLoader)

        except Exception as e:
            logging.error(e)
            sys.exit()

    def parse_params(self, params):
        #print(params) 
        print(f"grid size = {params['grid_size']}")
        try:
            
            self.n_fusedSilica = params['n_fusedSilica']
            self.n_PDMS = params['n_PDMS']
            self.n_amorphousSi = params['n_amorphousSi']

            self.pml_thickness = params['pml_thickness']
            self.height_pillar = params['height_pillar']
            self._radius = params['radius']
            self.rad_min = params['rad_min']
            self.rad_max = params['rad_max']
            self.width_PDMS = params['width_PDMS']
            self.width_fusedSilica = params['width_fusedSilica']
            self.non_pml = params['non_pml']
            
            self.center_PDMS = params['center_PDMS']
            self.center_fusedSilica = params['center_fusedSilica']
            self.center_pillar = params['center_pillar']
            
            self.z_fusedSilica = params['z_fusedSilica']
            self.z_PDMS = params['z_PDMS']
            self._x_dim = params['x_dim']
            self._y_dim = params['y_dim']
            self._geometry = params['geometry']
            
            self.wavelength_1550 = params['wavelength_1550']
            self.wavelength_1060 = params['wavelength_1060']
            self.wavelength_1300 = params['wavelength_1300']
            self.wavelength_1650 = params['wavelength_1650']
            self.wavelength_2881 = params['wavelength_2881']
            self.freq_1550 = params['freq_1550']
            self.freq_1060 = params['freq_1060']
            self.freq_1650 = params['freq_1650']
            self.freq_2881 = params['freq_2881']
            #self.lambda_min = params['lambda_min']
            #self.lambda_max = params['lambda_max']
            #self.f_min = params['f_min']
            #self.f_max = params['f_max']
            self.fcen = params['fcen']
            self.fwidth = params['fwidth']
            self.k_point = params['k_point']
            self.center_source = params['center_source']
            self.source_cmpt = params['source_cmpt']
            self._source_type = params['source_type']
            self._decay_rate = params['decay_rate']
            self._source = params['source']
            
            self.resolution = params['resolution']
            self.lattice_size = params['lattice_size']
            self._grid_size = params['grid_size']
            self.cell_x = params['cell_x']
            self.cell_y = params['cell_y']
            self.cell_z = params['cell_z']
            self.cell_size = params['cell_size']
            self.pml_layers = params['pml_layers']
            self._symmetries = params['symmetries']

            self.nfreq = params['nfreq']
            self.df = params['df']
            self.fr_center = params['fr_center']
            self._fr = params['fr']
            self.freq_list = params['freq_list']
 
            self.plot_plane = params['plot_plane']
            self.fps = params['fps']

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            sys.exit()
    
    def calculate_dependencies(self):
        
        self.cell_z = round(2 * self.pml_thickness + self.width_PDMS + self.height_pillar +  self.width_fusedSilica, 3)
        self.center_PDMS = round(0.5*(self.height_pillar + self.width_PDMS + self.pml_thickness) + (self.pml_thickness + self.width_fusedSilica) - 0.5 * self.cell_z, 3) 
        self.center_fusedSilica = round(0.5 * (self.pml_thickness + self.width_fusedSilica) - 0.5 * self.cell_z, 3)
        self.center_pillar = round(self.pml_thickness + self.width_fusedSilica + 0.5 * self.height_pillar- 0.5 * self.cell_z, 3)  
        self.z_fusedSilica = self.pml_thickness + self.width_fusedSilica
        self.z_PDMS = self.height_pillar + self.width_PDMS + self.pml_thickness
        self.non_pml = self.cell_z - (2 * self.pml_thickness)
 
        #self.f_min = 1 / self.lambda_max
        #self.f_max = 1 / self.lambda_min
        #self.fcen = 0.5 * (self.f_min + self.f_max)
        self.freq_1550 = 1 / self.wavelength_1550
        self.freq_1060 = 1 / self.wavelength_1060
        self.freq_1300 = 1 / self.wavelength_1300
        self.freq_1650 = 1 / self.wavelength_1650
        self.fcen = 1 / self.wavelength_1550
        self.freq_2881 = self.fcen - (self.freq_1060 - self.fcen)
        self.wavelength_2881 = 1 / self.freq_2881
        self.fwidth = 1.2 * self.fcen

        self.k_point = mp.Vector3(0, 0, 0)
        self.center_source = round(self.pml_thickness + self.width_fusedSilica * 0.2 - 0.5 * self.cell_z, 3) 
        self.source_cmpt = mp.Ey
        self._symmetries = self.symmetries
        
        self.cell_x = self.lattice_size * self._grid_size
        self.cell_y = self.lattice_size * self._grid_size
        self.cell_size = mp.Vector3(self.cell_x, self.cell_y, self.cell_z)
        self.pml_layers = [mp.PML(thickness = self.pml_thickness, direction = mp.Z)]
        
        self.fr_center = round(0.5 * self.cell_z - self.pml_thickness - 0.3 * self.width_PDMS, 3)
        self._fr = mp.FluxRegion(center = mp.Vector3(0,0,self.fr_center),
                                    size=mp.Vector3(self.cell_x, self.cell_y, 0))
        self.plot_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(self.lattice_size, 0, self.cell_z))    
        self.near_pt = mp.Vector3(0, 0, self.fr_center)
        self.near_vol = mp.Volume(center = mp.Vector3(0,0,0),
                                    size = mp.Vector3(self.cell_x, self.cell_y, self.non_pml))
        self.freq_list = [self.freq_2881, self.freq_1650, self.freq_1550, self.freq_1300, self.freq_1060]

        self.collect_params()    

    def collect_params(self):
        
        self._material_params = {
                            'n_fusedSilica'         : self.n_fusedSilica,
                            'n_PDMS'                : self.n_PDMS,
                            'n_amorphousSi'         : self.n_amorphousSi,
                            'pml_thickness'         : self.pml_thickness,
                            'height_pillar'         : self.height_pillar,
                            'radius'                : self._radius,
                            'rad_min'               : self.rad_min,
                            'rad_max'               : self.rad_max,
                            'width_PDMS'            : self.width_PDMS,
                            'width_fusedSilica'     : self.width_fusedSilica,
                            'center_PDMS'           : self.center_PDMS, 
                            'center_fusedSilica'    : self.center_fusedSilica,
                            'center_pillar'         : self.center_pillar,
                                            
                            'z_fusedSilica'         : self.z_fusedSilica,     
                            'z_PDMS'                : self.z_PDMS,    
                            'x_dim'                 : self._x_dim,
                            'y_dim'                 : self._y_dim,
                            'geometry'              : self._geometry,

                            }

        self._source_params = {

                            #'lambda_min'            : self.lambda_min,
                            #'lambda_max'            : self.lambda_max,
                            #'f_min'                 : self.f_min,
                            #'f_max'                 : self.f_max,
                            'wavelength_1550'       : self.wavelength_1550,
                            'fcen'                  : self.fcen,
                            'fwidth'                : self.fwidth,
                            'k_point'               : self.k_point,
                            'center_source'         : self.center_source, 
                            'source_cmpt'           : self.source_cmpt,   
                            'source_type'           : self._source_type,
                            'cell_x'                : self.cell_x,
                            'cell_y'                : self.cell_y,
                            'cell_z'                : self.cell_z,
                            'cell_size'             : self.cell_size,
                            'fr_center'             : self.fr_center,
                            'decay_rate'            : self._decay_rate,
                            'freq_list'             : self.freq_list
                            #'source'                : self._source,
                            

                            }

        self._sim_params = { 

                            'resolution'            : self.resolution,
                            'lattice_size'          : self.lattice_size,
                            
                            'cell_x'                : self.cell_x,
                            'cell_y'                : self.cell_y,
                            'cell_z'                : self.cell_z,
                            'cell_size'             : self.cell_size,
                            
                            'pml_layers'            : self.pml_layers,
                            'k_point'               : self.k_point,
                            'symmetries'            : self._symmetries,
                            'geometry'              : self._geometry,
                            }

        self._flux_params = {
                            
                            'freq_1550'             : self.freq_1550,
                            'freq_1060'             : self.freq_1060,
                            'freq_1300'             : self.freq_1300,
                            'freq_1650'             : self.freq_1650,
                            'freq_2881'             : self.freq_2881,
                            'fcen'                  : self.fcen,                         
                            'nfreq'                 : self.nfreq,
                            'df'                    : self.df,
                            'freq_list'             : self.freq_list,
                            'fr_center'             : self.fr_center,
                            'fr'                    : self._fr,
                            #'flux_object'           : self._flux_object,
                            'center_source'         : self.center_source,
                            'cell_x'                : self.cell_x,
                            'cell_y'                : self.cell_y,
                            'near_pt'               : self.near_pt,
                            'near_vol'              : self.near_vol, 
                            }

        self._animation_params = {

                            'lattice_size'          : self.lattice_size,
                            'cell_z'                : self.cell_z,
                            'plot_plane'            : self.plot_plane,
                            'source_cmpt'           : self.source_cmpt,
                            'source_type'           : self._source_type,
                            'fr_center'             : self.fr_center,
                            'decay_rate'            : self._decay_rate,
                            'fps'                   : self.fps,
                            
                            }

        self._all_params =  {

                            'material_params'       : self._material_params,
                            'source_params'         : self._source_params,
                            'sim_params'            : self._sim_params,
                            'flux_params'           : self._flux_params,
                            'animation_params'      : self._animation_params,
                            }                   
        
    @property
    def fr(self):
        return self._source_type
            
    @fr.setter
    def fr(self, value):
        logging.debug("Parameter Manager | setting fr (flux region) to {}".format(value))
        self._fr = value
        self.calculate_dependencies()
 
    @property
    def source_type(self):
        return self._source_type
    
    @source_type.setter
    def source_type(self, value):
        logging.debug("Parameter Manager | setting source type to {}".format(value))
        self._source_type = value
        self.calculate_dependencies()

    @property
    def source(self):
        return self._source
    
    @source.setter
    def source(self, value):
        logging.debug("Parameter Manager | setting source object to {}".format(value))
        self._source = value
        self.calculate_dependencies()

    @property
    def grid_size(self):
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value):
        self._grid_size = value
        self.calculate_dependencies()
                     
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.calculate_dependencies()

    @property
    def x_dim(self):
        return self._x_dim

    @x_dim.setter
    def x_dim(self, value):
        self._x_dim = value
        self.calculate_dependencies()

    @property
    def y_dim(self):
        return self._y_dim

    @y_dim.setter
    def y_dim(self, value):
        self._y_dim = value
        self.calculate_dependencies()

    @property
    def decay_rate(self):
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, value):
        self._decay_rate = value
        self.calculate_dependencies()

    @property
    def material_params(self):
        return self._material_params

    @property
    def source_params(self):
        return self._source_params

    @property
    def sim_params(self):
        return self._sim_params

    @property
    def flux_params(self):
        return self._flux_params

    @property
    def animation_params(self):
        return self._animation_params

    @property
    def all_params(self):
        return self._all_params

    @property
    def symmetries(self):
        return self._symmetries

    @symmetries.setter
    def symmetries(self, value):
        if value == mp.Ey:
            self._symmetries = [mp.Mirror(mp.X, phase=+1), #epsilon has mirror symmetry in x and y, phase doesn't matter
                          mp.Mirror(mp.Y, phase=-1)] #but sources have -1 phase when reflected normal to their direction
        elif value == mp.Ex:                      #use of symmetries important here, significantly speeds up sim
            self._symmetries = [mp.Mirror(mp.X, phase=-1),
                          mp.Mirror(mp.Y, phase=+1)]
        elif value == mp.Ez:
            self._symmetries = [mp.Mirror(mp.X, phase=+1),
                          mp.Mirror(mp.Y, phase=+1)] 
        self.calculate_dependencies()

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        self._geometry = value
        self.calculate_dependencies()

if __name__=="__main__":
    params = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    pm = Parameter_Manager(params=params)
    print(pm.resolution)
