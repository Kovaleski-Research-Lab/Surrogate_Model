#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import yaml
import torch
import numpy as np
import segmentation_models_pytorch as smp

from copy import deepcopy as copy
from torchmetrics import PeakSignalNoiseRatio
from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn
#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append("../")

from utils import parameter_manager
from core import curvature
from core import propagator
from pytorch_lightning import LightningModule
from core import conv_upsample


class SurrogateModel(LightningModule):
    #def __init__(self, params_model, params_propagator):
    def __init__(self, params_model):
        super().__init__()
 
        self.params = params_model
        self.num_classes = int(self.params['num_classes'])
        self.learning_rate = self.params['learning_rate']
        self.transfer_learn = self.params['transfer_learn']
        self.backbone = self.params['backbone']
        self.weights = self.params['weights']
        self.data_shape = torch.tensor(self.params['data_shape'])
        self.checkpoint_path = self.params['path_checkpoint']
        self.freeze_encoder = self.params['freeze_encoder']

        self.mcl_params = self.params['mcl_params']
        self.alpha = self.mcl_params['alpha']
        #self.beta = self.mcl_params['beta']
        self.gamma = self.mcl_params['gamma']
        self.delta = self.mcl_params['delta']

        self.num_phase = 9

        self.select_model()
        #self.create_propagation_layer(params_propagator)

        # - Creating domain bounds for phase parameters
        #self.constrain_phase = torch.nn.Sigmoid()
        #self.constrain_derivative = torch.nn.Sigmoid()
        
        # - Initialize optimizer
        self.configure_optimizers()

        # I think there is a class of models that combine an inverse design dreamer with
        # a fully constrained physics decoder. A generator and a simulator.
        # Generative adversarial simulator?

        # Visualizing gradients in a yee lattice.

        #Lists to hold testing things - there might be a better way #TODO
        # for encoder
        self.val_phase_predictions = []
        self.val_phase_truth = []
        self.val_deriv_predictions = []
        self.val_deriv_truth = []

        self.train_phase_predictions = []
        self.train_phase_truth = []
        self.train_deriv_predictions = []
        self.train_deriv_truth = []

        # for resim -- these get populated at the very end. change this
        self.val_phase_pred_resim = []
        self.val_phase_truth_resim = []
        self.val_nf_pred_resim = []
        self.val_nf_truth_resim = []
        #self.val_ff_pred_resim = []
        #self.val_ff_truth_resim = []
    
        self.train_phase_pred_resim = []
        self.train_phase_truth_resim = []
        self.train_nf_pred_resim = []
        self.train_nf_truth_resim = []
        #self.train_ff_pred_resim = []
        #self.train_ff_truth_resim = []

        # for recon
        self.val_nf_amp_diff = []
        self.val_nf_angle_diff = []
        #self.val_ff_amp_diff = []
        #self.val_ff_angle_diff = []

        self.train_nf_amp_diff = []
        self.train_nf_angle_diff = []
        #self.train_ff_amp_diff = []
        #self.train_ff_angle_diff = []

        self.save_hyperparameters()

    def constrain_phase(self, phase): 
        return torch.sin(phase) * torch.pi  # first we constrain it by sin which is periodic
                                             # then we mult by pi to scale it

    def create_propagation_layer(self, params_propagator):
        self.prop = propagator.Propagator(params = params_propagator) 

    def select_model(self):
        #Model
        if self.backbone == "resnet18":
             model = smp.Unet(encoder_name = "resnet18", encoder_weights = self.weights, 
                         in_channels= self.data_shape[1], classes = 2)
        elif self.backbone == "resnet34":
            model = smp.Unet(encoder_name = "resnet34", encoder_weights = self.weights, 
                         in_channels= self.data_shape[1], classes = 2)
        elif self.backbone == "resnet50":
            model = smp.Unet(encoder_name = "resnet50", encoder_weights = self.weights, 
                         in_channels= self.data_shape[1], classes = 2)

        #model = model.float()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.seg_head = model.segmentation_head # encoder - decoder - segmentation head (takes output shape of decoder and makes a final decision. probs an MLP)

        # - Create first and last layer of unet to handle non-expected inputs

        # Using off the shelf models with custom simulated data poses a problem for everyone.
        # Here, Charlie  makes a whole layer to map the output of the simulation into a good shape
        # for the model. However, what if the outputs of the simulation was already a good shape?
        # Is there a new classification of models that are aimed at doing a good mapping from 
        # simulations to model inputs? Essentially an 'add on' to the front of your model to
        # preprocess simulated data. This might just be the definition of feature extraction.
       
        spatial_shape = tuple(self.data_shape[-2:].numpy()) 

        self.first = conv_upsample.get_conv_transpose(input_size = spatial_shape,
                                        in_channels = self.data_shape[1], 
                                        out_channels = self.data_shape[1], mod_size = 32)

        #self.first = self.first.float()

        temp = torch.rand(tuple(self.data_shape.numpy()))
        response = self.first(temp)
        
        self.last = conv_upsample.get_conv(spatial_shape, response.shape)
        #self.last = self.last.float()

        # - Creating latent space between encoder and decoder of unet architecture
        # This is the place where we should constrain the model for physics. Knowledge discovery.
        # We have a thick feature extractor into an MLP, back through an MLP, and then into
        # the thich decoding stage. The goal of the first MLP is to map a set of features
        # from the electric field to the material paramters. It might make sense to constrain
        # the features right before this MLP to some basis set (modes). We can then pull out
        # information about the modes while still fulfilling the aim of inverse design.

        num_features = self.encoder(temp)[-1].view(-1).shape[0]    # what are latent space shapes? run encoder with an example input and figure out the number of features. insert an MLP (nn.seq) - take that output and converts it to the shape of the latent space
        self.encode_phase = torch.nn.Sequential(torch.nn.Linear(num_features, 512),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(512, self.num_phase))  # num_phases sets the size of the latent space

        #self.encode_modes... we would take the modes and feed them into the phases (so you have two MLPs in succession) - each output of the MLPs means somethign different

        self.decode_phase = torch.nn.Sequential(torch.nn.Linear(self.num_phase, 512), 
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(512, num_features))

        #self.decode_modes... might take phases back up to mode size 

        # this lets us do alternating optimization - particularly, we are in the decoder stage. learning is halted for both phase and deriv. (turning grads off, not the layers. still making predictions)
        # we are making a forward pass, and we are making a partial backward pass. it's partial because learning is only on for the decoder.
        if self.freeze_encoder:
            for p in self.encoder.parameters():  # for every param in encoder, set grads to false. (turn off learning for the encoder)
                p.requires_grad = False
            for p in self.first.parameters():  # this is the "first" layer - se thtose to false too.
                p.requires_grad = False
            for p in self.encode_phase.parameters():   # set all the params in the MLP to false. also turn off learnign for the MLP.
                p.requires_grad = False

    def ae_loss(self, preds, labels, choice): # ae=autoencoder. this chooses what loss func we use (used for other stuff besides ae) we need to add earth mover's distance
        if(choice == 0):
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif(choice == 1):
            fn = PeakSignalNoiseRatio()
            loss = 1 / (1 + fn(preds, labels))
        else:
            pass # make it earth movers distance 
        return loss

    #def objective(self, batch, predictions, alpha = 1, beta = 1, gamma = 1, delta = 1):
    def objective(self, batch, predictions, alpha = 1, gamma = 1, delta = 1):

        #near_fields, far_fields, radii, phases, derivatives = batch
        near_fields, radii, phases, derivatives = batch

        near_fields = near_fields[:,1,:,:,:].float().squeeze() # 1=y component
        #far_fields = far_fields[:,1,:,:,:].float().squeeze()
        radii = radii.squeeze()
        phases = phases.squeeze()
        derivatives = derivatives.squeeze()

        #pred_near_fields, pred_far_fields, pred_phases, pred_derivatives = predictions
        pred_near_fields, pred_phases, pred_derivatives = predictions

        #Training: Phase, Curvature, Efield
        
        near_field_loss = self.ae_loss(pred_near_fields.squeeze(), near_fields, choice = 0) # use EMV eventually (can use anything)
        #far_field_loss = self.ae_loss(pred_far_fields.squeeze(), far_fields, choice = 0)
        phase_loss = self.ae_loss(pred_phases.squeeze(), phases, choice = 0) # stick with MSE for this
        derivative_loss = self.ae_loss(pred_derivatives.squeeze(), derivatives, choice = 0) # stick with MSE for this

        #total_loss = self.alpha*near_field_loss + self.beta*far_field_loss + self.gamma*phase_loss + self.delta*derivative_loss
        total_loss = self.alpha*near_field_loss + self.gamma*phase_loss + self.delta*derivative_loss
    
       # return {"total_loss": total_loss, "near_field_loss": near_field_loss, 
       #         "phase_loss": phase_loss, "far_field_loss" : far_field_loss,
       #         "derivative_loss": derivative_loss}
        return {"total_loss": total_loss, "near_field_loss": near_field_loss, 
                "phase_loss": phase_loss, "derivative_loss": derivative_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    #------------------------------
    # Convert: Phase To Derivatives
    #------------------------------
    def convert_phase(self, batch_phase):
        batch_size = batch_phase.size()[0]
        return curvature.get_der_train(batch_phase.view(batch_size, 3, 3))
    
    def get_abs_difference(self, prediction, label):

        return torch.abs(prediction - label)

    def organize_testing(self, predictions, batch, batch_idx, dataloader): # this is part of those lists that has to be fixed.
        #pred_near_field, pred_far_field, pred_phase, pred_derivative = predictions
        pred_near_field, pred_phase, pred_derivative = predictions
        #true_near_field, true_far_field, true_radii, true_phase, true_derivative = batch
        true_near_field, true_radii, true_phase, true_derivative = batch
       
        if dataloader == 0: #Val dataloader 
            # encoder
            self.val_phase_predictions.append(pred_phase.detach().cpu().numpy())
            self.val_phase_truth.append(true_phase.detach().cpu().numpy())
            self.val_deriv_predictions.append(pred_derivative.detach().cpu().numpy())
            self.val_deriv_truth.append(true_derivative.detach().cpu().numpy())
            
            # decoder
            self.val_nf_amp_diff.append(self.get_abs_difference(pred_near_field[:,0,:,:], true_near_field[:,1,0,:,:]).detach().cpu().numpy())
            self.val_nf_angle_diff.append(self.get_abs_difference(pred_near_field[:,1,:,:], true_near_field[:,1,1,:,:]).detach().cpu().numpy())

            #self.val_ff_amp_diff.append(self.get_abs_difference(pred_far_field[:,0,:,:], true_far_field[:,1,0,:,:]).detach().cpu().numpy())
            #self.val_ff_angle_diff.append(self.get_abs_difference(pred_far_field[:,1,:,:], true_far_field[:,1,1,:,:]).detach().cpu().numpy())

            # resim
            if batch_idx == 0:
                self.val_phase_pred_resim.append(pred_phase.detach().cpu().numpy())
                self.val_phase_truth_resim.append(true_phase.detach().cpu().numpy())

                self.val_nf_pred_resim.append(pred_near_field.detach().cpu().numpy())
                self.val_nf_truth_resim.append(true_near_field[:,1,:,:,:].detach().cpu().numpy())

                #self.val_ff_pred_resim.append(pred_far_field.detach().cpu().numpy())
                #self.val_ff_truth_resim.append(true_far_field[:,1,:,:,:].detach().cpu().numpy())
                
        elif dataloader == 1: #Train dataloader
            # encoder
            self.train_phase_predictions.append(pred_phase.detach().cpu().numpy())
            self.train_phase_truth.append(true_phase.detach().cpu().numpy())

            self.train_deriv_predictions.append(pred_derivative.detach().cpu().numpy())
            self.train_deriv_truth.append(true_derivative.detach().cpu().numpy())

            # decoder
            self.train_nf_amp_diff.append(self.get_abs_difference(pred_near_field[:,0,:,:], true_near_field[:,1,0,:,:]).detach().cpu().numpy())
            self.train_nf_angle_diff.append(self.get_abs_difference(pred_near_field[:,1,:,:], true_near_field[:,1,1,:,:]).detach().cpu().numpy())

            #self.train_ff_amp_diff.append(self.get_abs_difference(pred_far_field[:,0,:,:], true_far_field[:,1,0,:,:]).detach().cpu().numpy())
            #self.train_ff_angle_diff.append(self.get_abs_difference(pred_far_field[:,1,:,:], true_far_field[:,1,1,:,:]).detach().cpu().numpy())

            # resim
            if batch_idx == 0:
                self.train_phase_pred_resim.append(pred_phase.detach().cpu().numpy())
                self.train_phase_truth_resim.append(true_phase.detach().cpu().numpy())

                self.train_nf_pred_resim.append(pred_near_field.detach().cpu().numpy())
                self.train_nf_truth_resim.append(true_near_field[:,1,:,:,:].detach().cpu().numpy())

                #self.train_ff_pred_resim.append(pred_far_field.detach().cpu().numpy())
                #self.train_ff_truth_resim.append(true_far_field[:,1,:,:,:].detach().cpu().numpy())

        else:
            exit()

    def forward(self, x):

        # Encoder: Feature Reduction
        x = self.first(x)
        x = self.encoder(x)

        # - Get last layer from Resnet encoder
        x_last = x[-1]
        x_shape = x_last.size()
        x_last = x_last.view(x_shape[0], -1)

        # Learning: Phase parameters, derivatives
        phase = self.encode_phase(x_last)

        # Constrain phase
        phase = self.constrain_phase(phase)
       
        # do you calculate derivatives before or after constraining phase? I think after.  
        derivatives = self.convert_phase(phase)

        # Decoder: Feature Reconstruction
        x_last = self.decode_phase(phase) # MLP 
        x_last = x_last.view(x_shape) 

        # - Update last layer from Resnet encoder
        x[-1] = x_last
        x_recon = self.seg_head(self.decoder(*x)) # MLP for final decison - takes output of the decoder and makes "classification" predictions. in our case, we're using it to give us amplitude and phase. we're co-opting a segmentation model to do this.
        # seg head gives us a weird shape: (batch, channel, width, head) - so we send it to our self.last()
        x_recon = self.last(x_recon) # gets reshaped

        return x_recon, phase, derivatives

    def shared_step(self, batch, batch_idx): # training step, valid step, and testing all call this function. 
        #near_fields, far_fields, radii, phases, derivatives = batch
        near_fields, radii, phases, derivatives = batch
        
        near_fields = near_fields.to(self.device)
        #far_fields = far_fields.to(self.device)
        radii = radii.to(self.device)
        phases = phases.to(self.device)
        derivatives = derivatives.to(self.device)
           
        #Going to just use the y component for now
        near_fields = near_fields[:,1,:,:,:].float()
        #far_fields = far_fields[:,1,:,:,:].float()
        #from IPython import embed; embed(); exit()
        pred_near_field, pred_phase, pred_derivatives = self.forward(near_fields)

        #wavefront = pred_near_field[:,0,:,:] * torch.exp(1j*pred_near_field[:,1,:,:]) # this is the near field wavefront.

        # propagate to the far field
        #pred_far_field = self.prop(wavefront)
        #pred_far_field = torch.cat((pred_far_field.abs().unsqueeze(dim=1).float(), pred_far_field.angle().unsqueeze(dim=1).float()), dim=1) 

        #return pred_near_field, pred_far_field, pred_phase, pred_derivatives
        return pred_near_field, pred_phase, pred_derivatives
        
    def training_step(self, batch, batch_idx):
        #Get predictions
        predictions = self.shared_step(batch, batch_idx)

        #Calculate loss
        loss = self.objective(batch, predictions)
        total_loss = loss['total_loss']
        near_field_loss = loss['near_field_loss']
        #far_field_loss = loss['far_field_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']
        
        #Log the loss
        self.log("train_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_near_field_loss", near_field_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        #self.log("train_far_field_loss", far_field_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}

    def validation_step(self, batch, batch_idx):
        #Get predictions
        predictions = self.shared_step(batch, batch_idx)

        #Calculate loss
        loss = self.objective(batch, predictions)
        total_loss = loss['total_loss']
        near_field_loss = loss['near_field_loss']
        #far_field_loss = loss['far_field_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']

        #Log the loss
        self.log("val_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_near_field_loss", near_field_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        #self.log("val_far_field_loss", far_field_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}
 
    def test_step(self, batch, batch_idx, dataloader_idx=0): # this lets us do evaluation
        #Get predictions
        predictions = self.shared_step(batch, batch_idx) # grabs all preds from the dataloader
        self.organize_testing(predictions, batch, batch_idx, dataloader_idx) # fills in those lists we defined above.

    def on_test_end(self): 

        # Encoder
        train_encoder = {
            'phase_truth': np.concatenate(self.train_phase_truth),
            'phase_pred' : np.concatenate(self.train_phase_predictions),
            'deriv_truth': np.concatenate(self.train_deriv_truth),
            'deriv_pred' : np.concatenate(self.train_deriv_predictions),
            }
        val_encoder = {
            'phase_truth': np.concatenate(self.val_phase_truth),
            'phase_pred' : np.concatenate(self.val_phase_predictions),
            'deriv_truth': np.concatenate(self.val_deriv_truth),
            'deriv_pred' : np.concatenate(self.val_deriv_predictions),
            }

        # Resim
        train_resim = {
            'phase_pred'    : np.concatenate(self.train_phase_pred_resim),
            'phase_truth'   : np.concatenate(self.train_phase_truth_resim),
            'nf_pred'       : np.concatenate(self.train_nf_pred_resim),
            'nf_truth'      : np.concatenate(self.train_nf_truth_resim),
            #'ff_pred'       : np.concatenate(self.train_ff_pred_resim),
            #'ff_truth'      : np.concatenate(self.train_ff_truth_resim), 
            }

        val_resim = {
            'phase_pred'    : np.concatenate(self.val_phase_pred_resim),
            'phase_truth'   : np.concatenate(self.val_phase_truth_resim),
            'nf_pred'       : np.concatenate(self.val_nf_pred_resim),
            'nf_truth'      : np.concatenate(self.val_nf_truth_resim),
            #'ff_pred'       : np.concatenate(self.val_ff_pred_resim),
            #'ff_truth'      : np.concatenate(self.val_ff_truth_resim), 
            }

        train_recon = {
            'nf_amp_diff'       : np.concatenate(self.train_nf_amp_diff),
            'nf_angle_diff'     : np.concatenate(self.train_nf_angle_diff),
            #'ff_amp_diff'       : np.concatenate(self.train_ff_amp_diff),
            #'ff_angle_diff'     : np.concatenate(self.train_ff_angle_diff),
            }
        val_recon = {
            'nf_amp_diff'       : np.concatenate(self.val_nf_amp_diff),
            'nf_angle_diff'     : np.concatenate(self.val_nf_angle_diff),
            #'ff_amp_diff'       : np.concatenate(self.val_ff_amp_diff),
            #'ff_angle_diff'     : np.concatenate(self.val_ff_angle_diff),
            }        
        #log_results(self, results, epoch, mode, count = 5, name = None):       
        self.logger.experiment.log_results(results = train_encoder, epoch=None, count=5, mode = 'train', name='encoder')
        self.logger.experiment.log_results(results = train_resim, epoch=None, count=5, mode = 'train', name='resim')
        self.logger.experiment.log_results(results = train_recon, epoch=None, count=5, mode = 'train', name='recon')

        self.logger.experiment.log_results(results = val_encoder, epoch=None, count=5, mode = 'val', name='encoder')
        self.logger.experiment.log_results(results = val_resim, epoch=None, count=5, mode = 'val', name='resim')
        self.logger.experiment.log_results(results = val_recon, epoch=None, count=5, mode = 'val', name='recon')




class Encoder(LightningModule):
    def __init__(self, params_model):
        super().__init__()
        self.params = params_model
        self.num_classes = int(self.params['num_classes'])
        self.learning_rate = self.params['learning_rate']
        self.transfer_learn = self.params['transfer_learn']
        self.backbone = self.params['backbone']
        self.weights = self.params['weights']
        self.data_shape = torch.tensor(self.params['data_shape'])
        self.checkpoint_path = self.params['path_checkpoint']

        self.mcl_params = self.params['mcl_params']
        self.alpha = self.mcl_params['alpha']
        #self.beta = self.mcl_params['beta']
        self.gamma = self.mcl_params['gamma']
        self.delta = self.mcl_params['delta']

        self.num_phase = 9

        self.select_model()
        
        # - Initialize optimizer
        self.configure_optimizers()

        self.save_hyperparameters()

    def select_model(self):
        if self.backbone == 'resnet50':
            backbone = resnet50(weights = self.weights)
        elif self.backbone == 'resnet34':
            backbone = resnet34(weights = self.weights)
        elif self.backbone == 'resnet18':
            backbone = resnet18(weights = self.weights)

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters,  self.num_phase)
        
        spatial_shape = tuple(self.data_shape[-2:].numpy()) 
        self.first = conv_upsample.get_conv_transpose(input_size = spatial_shape, in_channels = self.data_shape[1],
                                        out_channels = 3, mod_size = 32)
 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def ae_loss(self, preds, labels, choice):
        if(choice == 0):
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        else:
            fn = PeakSignalNoiseRatio()
            loss = 1 / (1 + fn(preds, labels))

        return loss

    #def objective(self, batch, predictions, alpha = 1, beta = 1, gamma = 1, delta = 1):
    def objective(self, batch, predictions, alpha = 1, gamma = 1, delta = 1):

        #near_fields, far_fields, radii, phases, derivatives = batch
        near_fields, radii, phases, derivatives = batch

        near_fields = near_fields[:,1,:,:,:].float().squeeze()
        #far_fields = far_fields[:,1,:,:,:].float().squeeze()
        radii = radii.squeeze()
        phases = phases.squeeze()
        derivatives = derivatives.squeeze()

        pred_phases, pred_derivatives = predictions

        #Training: Phase, Curvature, Efield
        
        phase_loss = self.ae_loss(pred_phases.squeeze(), phases, choice = 0)
        derivative_loss = self.ae_loss(pred_derivatives.squeeze(), derivatives, choice = 0)

        total_loss = self.gamma*phase_loss + self.delta*derivative_loss
    
        return {"total_loss": total_loss,
                "phase_loss": phase_loss,
                "derivative_loss": derivative_loss}
    
    def convert_phase(self, batch_phase):
        batch_size = batch_phase.size()[0]
        return curvature.get_der_train(batch_phase.view(batch_size, 3, 3))

    def shared_step(self, batch, batch_idx):
        #near_fields, far_fields, radii, phases, derivatives = batch
        near_fields, radii, phases, derivatives = batch
               
        #Going to just use the y component for now
        near_fields = near_fields[:,1,:,:,:].float()
        #far_fields = far_fields[:,1,:,:,:].float()

        pred_phase, pred_derivatives = self.forward(near_fields)
        return pred_phase, pred_derivatives

    def training_step(self, batch, batch_idx):
        #Get predictions
        predictions = self.shared_step(batch, batch_idx)

        #Calculate loss
        loss = self.objective(batch, predictions)
        total_loss = loss['total_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']
        
        #Log the loss
        self.log("train_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}

    def validation_step(self, batch, batch_idx):
        #Get predictions
        predictions = self.shared_step(batch, batch_idx)

        #Calculate loss
        loss = self.objective(batch, predictions)
        total_loss = loss['total_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']

        #Log the loss
        self.log("val_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}

    def test_step(self, batch, batch_idx):
        #Get predictions
        predictions = self.shared_step(batch, batch_idx)

        #Calculate loss
        loss = self.objective(batch, predictions)
        total_loss = loss['total_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']

        #Log the loss
        self.log("test_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("test_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("test_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}

  
if __name__ == "__main__":
    
    from IPython import embed;
    from pytorch_lightning import seed_everything
    from core import datamodule
    seed_everything(1337)
    params = yaml.load(open('../config.yaml', 'r'), Loader = yaml.FullLoader)
    pm = parameter_manager.Parameter_Manager(params = params)

    dm = datamodule.select_data(pm.params_datamodule)
    dm.prepare_data()
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    #model = CAI_Model(pm.params_model, pm.params_propagator)
    model = Encoder(pm.params_model)
    #embed()
    
