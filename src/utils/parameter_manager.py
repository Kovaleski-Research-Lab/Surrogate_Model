import meep as mp
import logging
import yaml
import sys
import traceback
from IPython import embed
import logging
import torch
import traceback
import os

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
        try:
            self._resim = params['resim']
            self._training_stage = params['training_stage']
            # ML params first:
            # Load: Paths 
            self.path_root = params['path_root']
            self.path_data = params['path_data']
            #self.path_model = params['path_model']
            self.path_train = params['path_train']
            self.path_valid = params['path_valid']
            self.path_results = params['path_results']
            self.path_resims = params['path_resims']
            #self._path_checkpoint = params['path_checkpoint']
 
            # Load: Trainer Params
            self.batch_size = params['batch_size']
            self.num_epochs = params['num_epochs']
            self.valid_rate = params['valid_rate']
            self.accelerator = params['accelerator']
            self.gpu_flag, self.gpu_list = params['gpu_config']

            # Load: Model Params
            self.weights = params['weights']
            self.backbone = params['backbone']
            self.optimizer = params['optimizer']
            self._mcl_params = params['mcl_params']
            self.initial_intensities = params['initial_intensities']
            self.num_classes = params['num_classes']
            self.learning_rate = params['learning_rate']
            self.transfer_learn = params['transfer_learn']
            self.freeze_encoder = params['freeze_encoder']
            self.load_checkpoint = params['load_checkpoint']
            self.objective_function = params['objective_function']

            self.initial_intensities = params['initial_intensities']

            # Load: Datamodule Params
            self._which = params['which']
            self._source_wl = params['source_wl']
            self.n_cpus = params['n_cpus']            
           
            # Determine the type of experiment we are running
            self.exp_name = params['exp_name']
            self.model_id = params['model_id']
            self.prev_model_id = params['prev_model_id']
         
            self.path_results = f"{self.path_results}/{self.model_id}/"
            self.seed_flag, self.seed_value = params['seed']

            # Now, datagen params:   
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
           
            self.wavelengths = params['wavelengths'] 
            self.cen_wavelength = params['cen_wavelength']
            #self.wavelength_1550 = params['wavelength_1550']
            #self.wavelength_1060 = params['wavelength_1060']
            #self.wavelength_1300 = params['wavelength_1300']
            #self.wavelength_1650 = params['wavelength_1650']
            #self.wavelength_2881 = params['wavelength_2881']
            #self.freq_1550 = params['freq_1550']
            #self.freq_1060 = params['freq_1060']
            #self.freq_1650 = params['freq_1650']
            #self.freq_2881 = params['freq_2881']
            #self.freq_1300 = params['freq_1300']
            self.fcen = params['fcen']
            self.fwidth = params['fwidth']

            self.k_point = params['k_point']
            self.center_source = params['center_source']
            self.source_cmpt = params['source_cmpt']
            self._source_type = params['source_type']
            self._decay_rate = params['decay_rate']
            self._source = params['source']
            self.dt = params['dt']
            
            self.resolution = params['resolution']
            self.lattice_size = params['lattice_size']
            self._grid_size = params['grid_size']
            self.cell_x = params['cell_x']
            self.cell_y = params['cell_y']
            self.cell_z = params['cell_z']
            self.cell_size = params['cell_size']
            self.pml_layers = params['pml_layers']
            self._symmetries = params['symmetries']
            
            self.Nxp = params['Nxp']
            self.Nyp = params['Nyp']
    
             # Datashape from the sim information

            self._data_shape = params['data_shape'] 

            self.nfreq = params['nfreq']
            self.df = params['df']
            self.mon_center = params['mon_center']
            self.freq_list = params['freq_list']
            self.cs = params['cs']
 
            self.plot_plane = params['plot_plane']
            self.fps = params['fps']
            
            self.path_dataset = params['path_dataset']

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            sys.exit()

    def get_prev_name(self, name):
        try:
            number = int(name.split('_')[1])
        except (IndexError, ValueError):
            return None

        # Calculate prev_name based on the numeric part of name
        prev_number = number - 1
        if prev_number < 0:
            print("First stage of training. load_checkpoint must be false: {pm.load_checkpoint}")
            return ""
 
        # Construct prev_name
        prev_name = f"{self.exp_name}_{prev_number}"

        return prev_name

    def calculate_dependencies(self):
        
        self.cell_z = (round(2 * self.pml_thickness + self.width_PDMS + self.height_pillar + 
                            self.width_fusedSilica, 3))
        self.center_PDMS = (round(0.5*(self.height_pillar + self.width_PDMS + self.pml_thickness)
                                + (self.pml_thickness + self.width_fusedSilica) - 0.5 * self.cell_z, 3)) 
        self.center_fusedSilica = (round(0.5 * (self.pml_thickness + self.width_fusedSilica) -
                                0.5 * self.cell_z, 3))
        self.center_pillar = (round(self.pml_thickness + self.width_fusedSilica +
                                0.5 * self.height_pillar- 0.5 * self.cell_z, 3))  
        self.z_fusedSilica = self.pml_thickness + self.width_fusedSilica
        self.z_PDMS = self.height_pillar + self.width_PDMS + self.pml_thickness
        self.non_pml = self.cell_z - (2 * self.pml_thickness)
 
        #self.freq_1550 = 1 / self.wavelength_1550
        #self.freq_1060 = 1 / self.wavelength_1060
        #self.freq_1300 = 1 / self.wavelength_1300
        #self.freq_1650 = 1 / self.wavelength_1650
        #self.freq_2881 = 1 / self.wavelength_2881
        self.fcen = 1 / self.cen_wavelength
        #self.freq_2881 = self.fcen - (self.freq_1060 - self.fcen)
        #self.wavelength_2881 = 1 / self.freq_2881
        self.fwidth = 1.2 * self.fcen

        self.k_point = mp.Vector3(0, 0, 0)
        self.center_source = (round(self.pml_thickness + self.width_fusedSilica * 0.2 -
                                0.5 * self.cell_z, 3))
        self.source_cmpt = mp.Ey
        self._symmetries = self.symmetries
        
        self.cell_x = self.lattice_size * self._grid_size
        self.cell_y = self.lattice_size * self._grid_size
        self.cell_size = mp.Vector3(self.cell_x, self.cell_y, self.cell_z)
        self.pml_layers = [mp.PML(thickness = self.pml_thickness, direction = mp.Z)]
        
        self.mon_center = round(0.5 * self.cell_z - self.pml_thickness - 0.3 * self.width_PDMS, 3)
        self.plot_plane = mp.Volume(center = mp.Vector3(0,0,0),
                                    size = mp.Vector3(self.lattice_size, 0, self.cell_z))    
        self.near_pt = mp.Vector3(0, 0, self.mon_center)
        self.near_vol = mp.Volume(center = mp.Vector3(0,0,0),
                                size = mp.Vector3(self.cell_x, self.cell_y, self.non_pml))
       
        self.freq_list = [ 1 / wl for wl in self.wavelengths]  
        #self.freq_list = [self.freq_2881, self.freq_1650, self.freq_1550, self.freq_1300, self.freq_1060]
        self.cs = [mp.Ex, mp.Ey, mp.Ez]
        self._data_shape = [1, 30, self.Nxp, self.Nyp]
        self.exp_name = self.model_id.split('_')[0]
        self.prev_model_id = self.get_prev_name(self.model_id) 
        self.collect_params()    

    def collect_params(self):
        logging.debug("Parameter_Manager | collecting parameters")

        # First, ML params:
        self._params_model = {
                                'weights'               : self.weights,
                                'backbone'              : self.backbone,
                                'optimizer'             : self.optimizer,
                                'data_shape'            : self._data_shape,
                                'num_classes'           : self.num_classes,
                                'learning_rate'         : self.learning_rate,
                                'transfer_learn'        : self.transfer_learn, 
                                #'path_checkpoint'       : self.path_checkpoint,
                                'load_checkpoint'       : self.load_checkpoint,
                                'objective_function'    : self.objective_function,
                                'mcl_params'            : self._mcl_params,
                                'initial_intensities'   : self.initial_intensities,
                                'freeze_encoder'        : self.freeze_encoder,
                                'freq_list'             : self.freq_list,
                                }
               
        self._params_datamodule = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'which'         : self._which,
                                'source_wl'     : self._source_wl,
                                'n_cpus'        : self.n_cpus,
                                'path_root'     : self.path_root, 
                                'path_data'     : self.path_data, 
                                'batch_size'    : self.batch_size, 
                                }

        self._params_trainer = {
                            'num_epochs'        : self.num_epochs, 
                            'valid_rate'        : self.valid_rate,
                            'accelerator'       : self.accelerator, 
                            }

        self._all_paths = {
                        'path_root'                     : self.path_root, 
                        'path_data'                     : self.path_data, 
                        #'path_model'                    : self.path_model,
                        'path_train'                    : self.path_train, 
                        'path_valid'                    : self.path_valid,
                        'path_results'                  : self.path_results, 
                        #'path_model'                    : self.path_model, 
                        #'path_results'                  : self.path_results, 
                        #'path_checkpoint'               : self._path_checkpoint,
                        'path_resims'                   : self.path_resims,
                        'path_dataset'                  : self.path_dataset, 
                        }

        # now, datagen paths:        
        self._geometry_params = {
                            'n_fusedSilica'         : self.n_fusedSilica,
                            'n_PDMS'                : self.n_PDMS,
                            'n_amorphousSi'         : self.n_amorphousSi,
                            'pml_thickness'         : self.pml_thickness,
                            'height_pillar'         : self.height_pillar,
                            'radius'                : self._radius,
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

                            'fcen'                  : self.fcen,
                            'fwidth'                : self.fwidth,
                            'center_source'         : self.center_source, 
                            'source_cmpt'           : self.source_cmpt,   
                            'source_type'           : self._source_type,
                            'cell_x'                : self.cell_x,
                            'cell_y'                : self.cell_y,
                            'cell_z'                : self.cell_z,
                            'decay_rate'            : self._decay_rate,
                            'source'                : self._source,

                            }

        self._sim_params = { 

                            'resolution'            : self.resolution,
                            'cell_x'                : self.cell_x,
                            'cell_y'                : self.cell_y,
                            'cell_z'                : self.cell_z,
                            'cell_size'             : self.cell_size,
                            'pml_layers'            : self.pml_layers,
                            'k_point'               : self.k_point,
                            'geometry'              : self._geometry,
                            'dt'                    : self.dt,
                            'source_cmpt'           : self.source_cmpt,
                            'mon_center'            : self.mon_center,
                            'decay_rate'            : self.decay_rate,
                            'source_type'           : self._source_type,
                            }

        self._dft_params = {
                                      
                            'freq_list'             : self.freq_list,
                            'near_vol'              : self.near_vol,
                            'cs'                    : self.cs, 

                            }

        self._animation_params = {

                            'lattice_size'          : self.lattice_size,
                            'cell_z'                : self.cell_z,
                            'plot_plane'            : self.plot_plane,
                            'source_cmpt'           : self.source_cmpt,
                            'source_type'           : self._source_type,
                            'mon_center'            : self.mon_center,
                            'decay_rate'            : self._decay_rate,
                            'fps'                   : self.fps,
                            
                            }

        self._all_params =  {
                            # datagen params:
                            'resim'                 : self._resim,
                            'training_stage'        : self._training_stage,
                            'geometry_params'       : self._geometry_params,
                            'source_params'         : self._source_params,
                            'sim_params'            : self._sim_params,
                            'dft_params'            : self._dft_params,
                            'animation_params'      : self._animation_params,
                            # ml params:
                            'model_params'          : self._params_model,
                            'datamodule_params'     : self._params_datamodule,
                            'all_paths'             : self._all_paths

                            }                   
    @property
    def training_stage(self):
        return self._training_stage

    @training_stage.setter
    def training_stage(self, value):
        self._training_stage = value
        self.calculate_dependencies()

    @property
    def resim(self):
        return self._resim

    @resim.setter
    def resim(self, value):
        self._resim = value
        self.calculate_dependencies()
        
    # ML getters/setters
    @property 
    def params_model(self):         
        return self._params_model

    @property
    def params_datamodule(self):
        return self._params_datamodule

    @property 
    def params_trainer(self):
        return self._params_trainer

    @property
    def all_paths(self):
        return self._all_paths 

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        logging.debug("Parameter_Manager | setting distance to {}".format(value))
        self._distance = value
        self.collect_params()
        
    #@property
    #def cen_wavelength(self):
    #    return self._cen_wavelength
    
    #@cen_wavelength.setter
    #def cen_wavelength(self, value):
    #    logging.debug("Parameter_Manager | setting center wavelength to {}".format(value))
    #    self._cen_wavelength = value
    #    self.collect_params()
    
    @property
    def data_shape(self):
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        logging.debug("Parameter_Manager | setting path_checkpoint to {}".format(value))
        self._data_shape = value
        self.collect_params()

    @property
    def which(self):
        return self._which

    @which.setter
    def which(self, value):
        logging.debug("Parameter_Manager | setting which to {}".format(value))
        self._which = value
        self.collect_params()

    @property
    def source_wl(self):
        return self._source_wl

    @source_wl.setter
    def source_wl(self, value):
        logging.debug("Parameter_Manager | setting source_wl to {}".format(value))
        self._source_wl = value
        self.collect_params()

    @property
    def adaptive(self):
        return self._adaptive

    @adaptive.setter
    def adaptive(self, value):
        logging.debug("Parameter_Manager | setting adaptive to {}".format(value))
        self._adaptive = value
        self.collect_params()

    @property
    def mcl_params(self):
        return self._mcl_params

    @mcl_params.setter
    def mcl_params(self, value):
        logging.debug("Parameter_Manager | setting mcl_params to {}".format(value))
        self._mcl_params = value
        self.collect_params()
 
    # datagen getters/setters
 
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
    def geometry_params(self):
        return self._geometry_params

    @property
    def source_params(self):
        return self._source_params

    @property
    def sim_params(self):
        return self._sim_params

    @property
    def dft_params(self):
        return self._dft_params

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
    pm = ParameterManager(params=params)
    print(pm.resolution)
    embed()
