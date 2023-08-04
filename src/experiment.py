import os 
import yaml
import torch
import logging
import argparse
import numpy as np

import train
from math import log10, floor
#import evaluate

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def begin_experiment(params):
    os.environ['TORCH_HOME'] = params['torch_home']
    #experiments = torch.load('utils/experiment_list_encoderTuning.pt')
    #

    #e = experiments[int(params['model_id'])]
    ##print(params)
    #for p in e:
    #    params[p] = e[p]
    #
    #params['model_id'] = "encoder_tuning_{:04d}".format(int(params['model_id']))
    #params['learning_rate'] = round_sig(params['learning_rate'])
    train.run(params)
    
    #params['mcl_params']['gamma'] = 1
    #params['learning_rate'] = 1.e-6
    #for i,e in enumerate(experiments):
    #    for p in e:
    #        params[p] = e[p]
    #    params['model_id'] = "encoderOnly_lrSweep_{:03d}".format(i) 
    #    train.run(params)

    #experiments = torch.load('utils/experiment_list_gamma.pt')

    #params['mcl_params']['gamma'] = 1
    #params['learning_rate'] = 1.e-6
    #for i,e in enumerate(experiments):
    #    for p in e:
    #        params[p] = e[p]
    #    params['model_id'] = "encoderOnly_gammaSweep_{:03d}".format(i) 
    #    train.run(params)
    #evaluate.run(params)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help = "Experiment: Train and Eval LRN Network")
    parser.add_argument("-which", help = "Which dataset to use")
    parser.add_argument("-objective_function", help = "Which objective function to train with")
    parser.add_argument("-transfer_learn", help = "Do you want to load in a pretrained model")
    parser.add_argument("-gpu_config", help = "Are you training with GPUs, and if so which ones")
    parser.add_argument("-num_epochs", help = "How many epochs to train for")
    parser.add_argument("-wavelength", help = "Which wavelength to run at?")
    parser.add_argument("-batch_size", help = "Batch size to use")
    parser.add_argument("-learning_rate", help = "learning rate")
    parser.add_argument("-job_id", help = "SLURM job ID")
    parser.add_argument("-model_id", help = "The line from hyperparameters.txt that you want to run")
    parser.add_argument("-distance", help = "Distance of propagation")

    args = parser.parse_args()
    if(args.config == None):
        logging.error("\nAttach Configuration File! Run experiment.py -h\n")
        exit()

    if args.job_id is not None:
        os.environ["SLURM_JOB_ID"] = args.job_id
        logging.debug("Slurm ID : {}".format(os.environ['SLURM_JOB_ID']))

    params = yaml.load(open(args.config), Loader = yaml.FullLoader)
   
    # Overwrite CLI specified parameters - Used for SLURM
    for k in params.keys():
        if k in args.__dict__ and args.__dict__[f'{k}'] is not None:
            params[f'{k}'] = args.__dict__[f'{k}']
            logging.debug("experiment.py | Setting {0} to {1}".format(k, args.__dict__[f'{k}']))

    begin_experiment(params)
