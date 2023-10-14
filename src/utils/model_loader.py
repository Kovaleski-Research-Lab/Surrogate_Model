#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from IPython import embed
#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core import datamodule, model


def select_model(pm):
    logging.debug("select_model.py - Selecting model") 
    network = None
    network = model.SurrogateModel(pm.params_model)

    # also consider loading in state of optimizer.
    if pm.load_checkpoint:
        folder_path = os.path.join(pm.path_root, "results/spie_journal_2023", pm.prev_model_id, "checkpoints")
        folder = os.listdir(folder_path)
        folder.sort()
        checkpoint = folder[-1]
        checkpoint_path = os.path.join(folder_path, checkpoint)
        state_dict = torch.load(checkpoint_path)['state_dict']
        network.load_state_dict(state_dict, strict=True)
    assert network is not None

    return network
