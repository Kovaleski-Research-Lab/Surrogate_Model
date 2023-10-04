#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import yaml
import torch
import numpy as np
import segmentation_models_pytorch as smp
from IPython import embed

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
from pytorch_lightning import LightningModule
from core import conv_upsample
from core import preprocess_data

class SurrogateModel(LightningModule):
    def __init__(self, params_model):
        super().__init__()
 
        self.params = params_model
        self.num_classes = int(self.params['num_classes'])
        self.learning_rate = self.params['learning_rate']
        self.transfer_learn = self.params['transfer_learn']
        self.backbone = self.params['backbone']
        self.weights = self.params['weights']
        self.data_shape = torch.tensor(self.params['data_shape'])
        #self.checkpoint_path = self.params['path_checkpoint']
        self.freeze_encoder = self.params['freeze_encoder']

        self.mcl_params = self.params['mcl_params']
        self.alpha = self.mcl_params['alpha'] # near field 
        #self.beta = self.mcl_params['beta'] # far field
        self.gamma = self.mcl_params['gamma'] # phase 
        self.delta = self.mcl_params['delta'] # derivatives
        self.epsilon = self.mcl_params['epsilon'] # source_flux
        self.initial_intensities = self.params['initial_intensities']

        self.num_phase = 9 

        self.select_model()

        # - Creating domain bounds for phase parameters
        
        # - Initialize optimizer
        self.configure_optimizers()

        # I think there is a class of models that combine an inverse design dreamer with
        # a fully constrained physics decoder. A generator and a simulator.
        # Generative adversarial simulator?

        # Visualizing gradients in a yee lattice.

        #Lists to hold testing things - there might be a better way #TODO
        # for encoder
        self.val_phase_pred, self.val_phase_truth = [], []
        self.val_deriv_pred, self.val_deriv_truth = [], []
        self.val_intensity_pred, self.val_intensity_truth = [], []

        self.train_phase_pred, self.train_phase_truth = [], []
        self.train_deriv_pred, self.train_deriv_truth = [], []    
        self.train_intensity_pred, self.train_intensity_truth = [], []

        # for resim -- these get populated at the very end. change this
        self.val_phase_pred_resim, self.val_phase_truth_resim = [], []
        self.val_intensity_pred_resim, self.val_intensity_truth_resim = [], []

        self.val_nf_2881_pred_resim, self.val_nf_2881_truth_resim = [], []
        self.val_nf_1650_pred_resim, self.val_nf_1650_truth_resim = [], []
        self.val_nf_1550_pred_resim, self.val_nf_1550_truth_resim = [], []
        self.val_nf_1300_pred_resim, self.val_nf_1300_truth_resim = [], []
        self.val_nf_1060_pred_resim, self.val_nf_1060_truth_resim = [], []
    
        self.train_phase_pred_resim, self.train_phase_truth_resim = [], []
        self.train_intensity_pred_resim, self.train_intensity_truth_resim = [], []

        self.train_nf_2881_pred_resim, self.train_nf_2881_truth_resim = [], []
        self.train_nf_1650_pred_resim, self.train_nf_1650_truth_resim = [], []
        self.train_nf_1550_pred_resim, self.train_nf_1550_truth_resim = [], []
        self.train_nf_1300_pred_resim, self.train_nf_1300_truth_resim = [], []
        self.train_nf_1060_pred_resim, self.train_nf_1060_truth_resim = [], []

        # for recon
        self.val_nf_2881_amp_diff, self.val_nf_2881_angle_diff = [], []
        self.val_nf_1650_amp_diff, self.val_nf_1650_angle_diff = [], []
        self.val_nf_1550_amp_diff, self.val_nf_1550_angle_diff = [], []
        self.val_nf_1300_amp_diff, self.val_nf_1300_angle_diff = [], []
        self.val_nf_1060_amp_diff, self.val_nf_1060_angle_diff = [], []

        self.train_nf_2881_amp_diff, self.train_nf_2881_angle_diff = [], []
        self.train_nf_1650_amp_diff, self.train_nf_1650_angle_diff = [], []
        self.train_nf_1550_amp_diff, self.train_nf_1550_angle_diff = [], []
        self.train_nf_1300_amp_diff, self.train_nf_1300_angle_diff = [], []
        self.train_nf_1060_amp_diff, self.train_nf_1060_angle_diff = [], []

        self.save_hyperparameters()

    def constrain_phase(self, phase): 
    
        #return (torch.sin(phase) * torch.pi).to(dtype=torch.float64)  # first we constrain it by sin which is periodic
        return (torch.sin(phase) * torch.pi)  # first we constrain it by sin which is periodic
                                             # then we mult by pi to scale it
       
    def select_model(self):

        #Model
        if self.backbone == "resnet18":
             model = smp.Unet(encoder_name = "resnet18", encoder_weights = self.weights, 
                         in_channels= self.data_shape[1], classes = 30)
        elif self.backbone == "resnet34":
            model = smp.Unet(encoder_name = "resnet34", encoder_weights = self.weights, 
                         in_channels= self.data_shape[1], classes = 30)
        elif self.backbone == "resnet50":
            model = smp.Unet(encoder_name = "resnet50", encoder_weights = self.weights, 
                         in_channels= self.data_shape[1], classes = 30)

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

        temp = torch.rand(tuple(self.data_shape.numpy()))
        response = self.first(temp)
        self.last = conv_upsample.get_conv(spatial_shape, response.shape)

        # - Creating latent space between encoder and decoder of unet architecture
        # This is the place where we should constrain the model for physics. Knowledge discovery.
        # We have a thick feature extractor into an MLP, back through an MLP, and then into
        # the thich decoding stage. The goal of the first MLP is to map a set of features
        # from the electric field to the material paramters. It might make sense to constrain
        # the features right before this MLP to some basis set (modes). We can then pull out
        # information about the modes while still fulfilling the aim of inverse design.
        
        num_features = self.encoder(temp)[-1].view(-1).shape[0]    # what are latent space shapes? run encoder with an example input and figure out the number of features. insert an MLP (nn.seq) - take that output and converts it to the shape of the latent space
        self.encode_phase = torch.nn.Sequential(torch.nn.Linear(num_features,512),
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

        elif(choice == 2): # this is thermo loss 
    
            # get initial intensities as a list in order: [2881...1060] use these as truth_initial and pred_initial
            i_initial = list(self.initial_intensities.values())
            i_initial = torch.tensor(i_initial, requires_grad=True).to(self.device)

            a = []                      # this is the first term of the loss function. a = abs(truth_final - truth_initial)
            i_final_truth = labels
            for val in i_final_truth:                   # iterate through each sample of the batch
                temp = []
                for i, truth_final in enumerate(val):   # get every frequency and calculate final-initial
                    temp.append(torch.abs(truth_final - i_initial[i]))
                a.append(temp)

            b = []                      # this is the second term of the loss function. b = abs(pred_final - pred_initial)
            i_final_pred = preds
            for val in i_final_pred:
                temp = []
                for i, pred_final in enumerate(val):
                    temp.append(torch.abs(pred_final - i_initial[i]))
                b.append(temp)
            
            loss = []
            for (A, B) in zip(a, b): # iterate over the length of a and b (so the batch)
                #a_i and b_i have length 5 for the loss terms associated with each frequency.
                temp = [torch.abs(x - y) for x,y in zip(A,B)]
                loss.append(temp)

            loss = torch.tensor(loss, requires_grad=True).to(self.device)
            agg_loss = torch.mean(loss, dim=0)  # this gives us a [5] tensor containing avg loss for each freq across the batch
            thermo_loss = torch.mean(agg_loss) # this aggregates all the frequencies so now we have a single loss term for thermo
            return thermo_loss
        else:
            pass # make it earth movers distance 
        return loss

    def get_mag_and_angle(self, field, comp_idx): # we use this to calculate near field loss

        comp = field[comp_idx,:,:,:]
        mag = comp[0,:,:]
        angle = comp[1,:,:]
        return mag, angle

    def calculate_intensity(self, x_mag, y_mag, z_mag):

        E_0 = torch.sqrt(torch.abs(x_mag)**2 + torch.abs(y_mag)**2 + torch.abs(z_mag)**2)
        intensity = 0.5 * E_0**2
        intensity = intensity.mean()
        return intensity

    def get_pred_intensities(self, pred_nf):

        pred_intensities = []

        for pred in pred_nf: # batch size is 8, so 8 iterations. shape of each pred_recon is [5, 3, 2, xdim, ydim]
            temp = [] # to hold the intensities of a particular wavelength
            for i in range(pred.shape[0]):
                freq = pred[i,:,:,:,:]
                x_mag = freq[0,0,:,:] # 0: x, 0: mag
                y_mag = freq[1,0,:,:] # 1: y, 0: mag
                z_mag = freq[2,0,:,:] # 2: z, 0: mag
                intensity = self.calculate_intensity(x_mag, y_mag, z_mag)
                temp.append(intensity)
            
            pred_intensities.append(temp) # index 0 holds sample 0, etc.

        pred_intensities = torch.tensor(pred_intensities, requires_grad=True) # matches the shape of intensities.
        
        return pred_intensities

    def calculate_thermo_loss(self):
        pass
 
    def objective(self, batch, predictions, all_nf, alpha = 1, gamma = 1, delta = 1, epsilon = 1):

        # We get truth values from the batch: phases, derivatives, intensities, near fields        
        radii = batch['radii'].squeeze()

        phases = batch['phases'].squeeze() 
        phases = self.constrain_phase(phases)

        derivatives = batch['derivatives'].squeeze()

        intensities = [batch['intensities_2881'], batch['intensities_1650'], batch['intensities_1550'], batch['intensities_1300'], batch['intensities_1060']]
        intensities = [x.squeeze() for x in intensities]
        intensities = torch.stack(intensities)
        intensities = intensities.transpose(0,1)

        nf = [batch['nf_2881'], batch['nf_1650'], batch['nf_1550'], batch['nf_1300'], batch['nf_1300']]
        nf = [x.squeeze() for x in nf]
        nf = torch.stack(nf)
        nf = nf.transpose(0, 1).float()

        # We get predictions from the model. We'll have to calculate predicted intensities.
        pred_nf, pred_phases, pred_derivatives = predictions[0], predictions[1], predictions[2]

        pred_intensities = self.get_pred_intensities(pred_nf)
        # this needs to be every mag and phase of every comp of every freq gets passed through MSE 
        # nf.shape and pred_nf.shape is [8, 5, 3, 2, 166, 166] - batch, freq, component, mag/angle, xdim, ydim

        x_loss, y_loss, z_loss = [], [], [] # append these list with shape [2] tensors: magnitude loss followed by angle loss
        for batch, (truths, preds) in enumerate(zip(nf, pred_nf)): 
            
            temp_x, temp_y, temp_z = [], [], []            
            for (t, p) in zip(truths, preds):
                # get the truth values
                x_mag, x_angle = self.get_mag_and_angle(t, 0)
                y_mag, y_angle = self.get_mag_and_angle(t, 1)
                z_mag, z_angle = self.get_mag_and_angle(t, 2)
                
                # get the preds
                x_mag_pred, x_angle_pred = self.get_mag_and_angle(p, 0)
                y_mag_pred, y_angle_pred = self.get_mag_and_angle(p, 1)
                z_mag_pred, z_angle_pred = self.get_mag_and_angle(p, 2)

                # There's a loss associated with magnitude and angle of each component.
                x_loss_mag = self.ae_loss(x_mag_pred, x_mag, choice=0)       
                x_loss_angle = self.ae_loss(x_angle_pred, x_angle, choice=0)
                temp_x.append(torch.cat((x_loss_mag.view(1), x_loss_angle.view(1)), dim=0))
    
                y_loss_mag = self.ae_loss(y_mag_pred, y_mag, choice=0)
                y_loss_angle = self.ae_loss(y_angle_pred, y_angle, choice=0)
                temp_y.append(torch.cat((y_loss_mag.view(1), y_loss_angle.view(1)), dim=0))
                
                z_loss_mag = self.ae_loss(z_mag_pred, z_mag, choice=0)
                z_loss_angle = self.ae_loss(z_angle_pred, z_angle, choice=0)
                temp_z.append(torch.cat((z_loss_mag.view(1), z_loss_angle.view(1)), dim=0))
          
            # for each batch, aggregate loss: index 0 is mag, index 1 is angle. 
            # whats the best way to aggregate loss, sum or mean? sum gives equal weight to each frequency's loss, mean loss treats all freq. equally in terms of the contribution to final loss.

            mean_loss_x_mag = sum([loss[0] for loss in temp_x]) / len(temp_x)
            mean_loss_x_angle = sum([loss[1] for loss in temp_x]) / len(temp_x)

            mean_loss_y_mag = sum([loss[0] for loss in temp_y]) / len(temp_y)
            mean_loss_y_angle = sum([loss[1] for loss in temp_y]) / len(temp_y)

            mean_loss_z_mag = sum([loss[0] for loss in temp_z]) / len(temp_z)
            mean_loss_z_angle = sum([loss[1] for loss in temp_z]) / len(temp_z)
            
            x_loss.append(mean_loss_x_mag + mean_loss_x_angle)
            y_loss.append(mean_loss_y_mag + mean_loss_y_angle)
            z_loss.append(mean_loss_z_mag + mean_loss_z_angle)

            # at the end of this loop, we have len({}_loss) = batch and gradients are preserved.
            # after exiting the loop we'll aggregate across the batch.
        
        x_loss = torch.stack(x_loss, dim=0)
        x_loss_agg = torch.mean(x_loss, dim=0).float()

        y_loss = torch.stack(y_loss, dim=0)
        y_loss_agg = torch.mean(y_loss, dim=0).float()

        z_loss = torch.stack(z_loss, dim=0)
        z_loss_agg = torch.mean(z_loss, dim=0).float()

        phase_loss = self.ae_loss(pred_phases.squeeze(), phases, choice = 0) # stick with MSE for this
        derivative_loss = self.ae_loss(pred_derivatives.squeeze(), derivatives, choice = 0) # stick with MSE for this
        thermo_loss = self.ae_loss(pred_intensities.squeeze(), intensities, choice = 2)
        
        total_loss = (self.alpha*x_loss_agg + self.alpha*y_loss_agg + self.alpha*z_loss_agg +
                            self.gamma*phase_loss + self.delta*derivative_loss + self.epsilon*thermo_loss).float()
        
        return {"x_loss": x_loss_agg, "y_loss": y_loss_agg, "z_loss": z_loss_agg,
                "phase_loss": phase_loss, "derivative_loss": derivative_loss,
                "thermo_loss": thermo_loss, "total_loss": total_loss} 

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
         
        pred_nf, pred_phases, pred_derivatives, pred_intensities = predictions[0], predictions[1], predictions[2], predictions[3]
        
        pred_nf_2881, true_nf_2881 = pred_nf[:,0,:,:,:,:], batch['nf_2881']
        pred_nf_1650, true_nf_1650 = pred_nf[:,1,:,:,:,:], batch['nf_1650']
        pred_nf_1550, true_nf_1550 = pred_nf[:,2,:,:,:,:], batch['nf_1550']
        pred_nf_1300, true_nf_1300 = pred_nf[:,3,:,:,:,:], batch['nf_1300']
        pred_nf_1060, true_nf_1060 = pred_nf[:,4,:,:,:,:], batch['nf_1060']
       
        true_phases, true_derivatives = batch['phases'], batch['derivatives']
        true_intensities = [batch['intensities_2881'], batch['intensities_1650'], batch['intensities_1550'],
                            batch['intensities_1300'], batch['intensities_1060']] 
        
        true_intensities = [x.detach().cpu() for x in true_intensities]
        true_intensities = [x.squeeze() for x in true_intensities]
        true_intensities = torch.stack(true_intensities)
        true_intensities = true_intensities.transpose(0,1)

        if dataloader == 0: #Val dataloader 
            # encoder
            self.val_phase_pred.append(pred_phases.detach().cpu().numpy())
            self.val_phase_truth.append(true_phases.detach().cpu().numpy())
            self.val_deriv_pred.append(pred_derivatives.detach().cpu().numpy())
            self.val_deriv_truth.append(true_derivatives.detach().cpu().numpy())
            self.val_intensity_pred.append(pred_intensities.detach().cpu().numpy())
            self.val_intensity_truth.append(true_intensities.numpy())

            # decoder (recon) - just doing y component for now.
            self.val_nf_2881_amp_diff.append(self.get_abs_difference(pred_nf_2881[:,1,0,:,:], # batch, component, mag/angle, xdim, ydim
                                                    true_nf_2881[:,1,0,:,:]).detach().cpu().numpy())
            self.val_nf_2881_angle_diff.append(self.get_abs_difference(pred_nf_2881[:,1,1,:,:],
                                                    true_nf_2881[:,1,1,:,:]).detach().cpu().numpy())

            self.val_nf_1650_amp_diff.append(self.get_abs_difference(pred_nf_1650[:,1,0,:,:],
                                                    true_nf_1650[:,1,0,:,:]).detach().cpu().numpy())
            self.val_nf_1650_angle_diff.append(self.get_abs_difference(pred_nf_1650[:,1,1,:,:],
                                                    true_nf_1650[:,1,1,:,:]).detach().cpu().numpy())

            self.val_nf_1550_amp_diff.append(self.get_abs_difference(pred_nf_1550[:,1,0,:,:],
                                                    true_nf_1550[:,1,0,:,:]).detach().cpu().numpy())
            self.val_nf_1550_angle_diff.append(self.get_abs_difference(pred_nf_1550[:,1,1,:,:],
                                                    true_nf_1550[:,1,1,:,:]).detach().cpu().numpy())

            self.val_nf_1300_amp_diff.append(self.get_abs_difference(pred_nf_1300[:,1,0,:,:],
                                                    true_nf_1300[:,1,0,:,:]).detach().cpu().numpy())
            self.val_nf_1300_angle_diff.append(self.get_abs_difference(pred_nf_1300[:,1,1,:,:],
                                                    true_nf_1300[:,1,1,:,:]).detach().cpu().numpy())

            self.val_nf_1060_amp_diff.append(self.get_abs_difference(pred_nf_1060[:,1,0,:,:],
                                                    true_nf_1060[:,1,0,:,:]).detach().cpu().numpy())
            self.val_nf_1060_angle_diff.append(self.get_abs_difference(pred_nf_1060[:,1,1,:,:],
                                                    true_nf_1060[:,1,1,:,:]).detach().cpu().numpy())
            
            #resim
            if batch_idx == 0:
                self.val_phase_pred_resim.append(pred_phases.detach().cpu().numpy())
                self.val_phase_truth_resim.append(true_phases.detach().cpu().numpy())
                self.val_intensity_pred_resim.append(pred_intensities.detach().cpu().numpy())
                self.val_intensity_truth_resim.append(true_intensities.detach().cpu().numpy())

                self.val_nf_2881_pred_resim.append(pred_nf_2881.detach().cpu().numpy())
                self.val_nf_2881_truth_resim.append(true_nf_2881[:,1,:,:,:].detach().cpu().numpy())
                
                self.val_nf_1650_pred_resim.append(pred_nf_1650.detach().cpu().numpy())
                self.val_nf_1650_truth_resim.append(true_nf_1650[:,1,:,:,:].detach().cpu().numpy())

                self.val_nf_1550_pred_resim.append(pred_nf_1550.detach().cpu().numpy())
                self.val_nf_1550_truth_resim.append(true_nf_1550[:,1,:,:,:].detach().cpu().numpy())
            
                self.val_nf_1300_pred_resim.append(pred_nf_1300.detach().cpu().numpy())
                self.val_nf_1300_truth_resim.append(true_nf_1300[:,1,:,:,:].detach().cpu().numpy())

                self.val_nf_1060_pred_resim.append(pred_nf_1060.detach().cpu().numpy())
                self.val_nf_1060_truth_resim.append(true_nf_1060[:,1,:,:,:].detach().cpu().numpy())
                

        elif dataloader == 1: #Train dataloader
            self.train_phase_pred.append(pred_phases.detach().cpu().numpy())
            self.train_phase_truth.append(true_phases.detach().cpu().numpy())
            self.train_deriv_pred.append(pred_derivatives.detach().cpu().numpy())
            self.train_deriv_truth.append(true_derivatives.detach().cpu().numpy())
            self.train_intensity_pred.append(pred_intensities.detach().cpu().numpy())
            self.train_intensity_truth.append(true_intensities.numpy())

            # decoder (recon)
            self.train_nf_2881_amp_diff.append(self.get_abs_difference(pred_nf_2881[:,1,0,:,:],
                                                 true_nf_2881[:,1,0,:,:]).detach().cpu().numpy())
            self.train_nf_2881_angle_diff.append(self.get_abs_difference(pred_nf_2881[:,1,1,:,:],
                                                 true_nf_2881[:,1,1,:,:]).detach().cpu().numpy())

            self.train_nf_1650_amp_diff.append(self.get_abs_difference(pred_nf_1650[:,1,0,:,:],
                                                 true_nf_1650[:,1,0,:,:]).detach().cpu().numpy())
            self.train_nf_1650_angle_diff.append(self.get_abs_difference(pred_nf_1650[:,1,1,:,:],
                                                 true_nf_1650[:,1,1,:,:]).detach().cpu().numpy())

            self.train_nf_1550_amp_diff.append(self.get_abs_difference(pred_nf_1550[:,1,0,:,:],
                                                 true_nf_1550[:,1,0,:,:]).detach().cpu().numpy())
            self.train_nf_1550_angle_diff.append(self.get_abs_difference(pred_nf_1550[:,1,1,:,:],
                                                 true_nf_1550[:,1,1,:,:]).detach().cpu().numpy())

            self.train_nf_1300_amp_diff.append(self.get_abs_difference(pred_nf_1300[:,1,0,:,:],
                                                 true_nf_1300[:,1,0,:,:]).detach().cpu().numpy())
            self.train_nf_1300_angle_diff.append(self.get_abs_difference(pred_nf_1300[:,1,1,:,:],
                                                 true_nf_1300[:,1,1,:,:]).detach().cpu().numpy())

            self.train_nf_1060_amp_diff.append(self.get_abs_difference(pred_nf_1060[:,1,0,:,:],
                                                 true_nf_1060[:,1,0,:,:]).detach().cpu().numpy())
            self.train_nf_1060_angle_diff.append(self.get_abs_difference(pred_nf_1060[:,1,1,:,:],
                                                    true_nf_1060[:,1,1,:,:]).detach().cpu().numpy())
            #resim
            if batch_idx == 0:
                self.train_phase_pred_resim.append(pred_phases.detach().cpu().numpy())
                self.train_phase_truth_resim.append(true_phases.detach().cpu().numpy())
                self.train_intensity_pred_resim.append(pred_intensities.detach().cpu().numpy())
                self.train_intensity_truth_resim.append(true_intensities.detach().cpu().numpy())

                self.train_nf_2881_pred_resim.append(pred_nf_2881.detach().cpu().numpy())
                self.train_nf_2881_truth_resim.append(true_nf_2881[:,1,:,:,:].detach().cpu().numpy())
                
                self.train_nf_1650_pred_resim.append(pred_nf_1650.detach().cpu().numpy())
                self.train_nf_1650_truth_resim.append(true_nf_1650[:,1,:,:,:].detach().cpu().numpy())

                self.train_nf_1550_pred_resim.append(pred_nf_1550.detach().cpu().numpy())
                self.train_nf_1550_truth_resim.append(true_nf_1550[:,1,:,:,:].detach().cpu().numpy())
            
                self.train_nf_1300_pred_resim.append(pred_nf_1300.detach().cpu().numpy())
                self.train_nf_1300_truth_resim.append(true_nf_1300[:,1,:,:,:].detach().cpu().numpy())

                self.train_nf_1060_pred_resim.append(pred_nf_1060.detach().cpu().numpy())
                self.train_nf_1060_truth_resim.append(true_nf_1060[:,1,:,:,:].detach().cpu().numpy())
        else:
            exit()

    def forward(self, x): # not exactly the forward pass. depending on if ptlightning called forward from training (grads) or valid (no grads) you ahve grads or not.
        # Encoder: Feature Reduction
        x = self.first(x.float())
        x = self.encoder(x)
        x_last = x[-1] 
        # - Get last layer from Resnet encoder
        x_shape = x_last.size()
        x_last = x_last.view(x_shape[0], -1)
        
        # Learning: Phase parameters, derivatives
        phase = self.encode_phase(x_last)

        # Constrain phase
        phase = self.constrain_phase(phase)
        # do you calculate derivatives or after constraining phase? I think after.  
        derivatives = self.convert_phase(phase)

        # Decoder: Feature Reconstruction
        x_last = self.decode_phase(phase) # MLP 
        
        x_last = x_last.view(x_shape) 
        # - Update last layer from Resnet encoder
        x[-1] = x_last
        x_recon = self.seg_head(self.decoder(*x)) # MLP for final decison - takes output of the decoder and makes "classification" predictions. in our case, we're using it to give us amplitude and phase. we're co-opting a segmentation model to do this.
        # seg head gives us a weird shape: (batch, channel, width, head) - so we send it to our self.last()
        recon = self.last(x_recon) # gets reshaped
        
        return [recon, phase, derivatives]

    def shared_step(self, batch, batch_idx): # training step, valid step, and testing all call this function. 

        nf_2881 = batch['nf_2881'].unsqueeze(dim=1)

        nf_1650 = batch['nf_1650'].unsqueeze(dim=1)

        nf_1550 = batch['nf_1550'].unsqueeze(dim=1)

        nf_1300 = batch['nf_1300'].unsqueeze(dim=1)

        nf_1060 = batch['nf_1060'].unsqueeze(dim=1)

        all_nf = torch.cat((nf_2881, nf_1650, nf_1550, nf_1300, nf_1060), dim=1).to(self.device)
        shape = all_nf.shape
        all_nf_reshaped = all_nf.view(shape[0], shape[1]*shape[2]*shape[3], shape[4], shape[5]) 
       
        outputs = self.forward(all_nf_reshaped)
        outputs[0] = outputs[0].view(shape) # reshaped back to same size as all_nf. now i can access components, frequencies, etc.

        return all_nf, outputs
        
    def training_step(self, batch, batch_idx):
        
        #Get predictions
        
        all_nf, predictions = self.shared_step(batch, batch_idx)
        #Calculate loss
        loss = self.objective(batch, predictions, all_nf)
        total_loss = loss['total_loss']
        x_loss = loss['x_loss']
        y_loss = loss['y_loss']
        z_loss = loss['z_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']
        thermo_loss = loss['thermo_loss']
        
        #Log the loss
        self.log("train_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_nf_x_loss", x_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_nf_y_loss", y_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_nf_z_loss", z_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_thermo_loss", thermo_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}

    def validation_step(self, batch, batch_idx):
        #Get predictions
        all_nf, predictions = self.shared_step(batch, batch_idx)

        #Calculate loss
        loss = self.objective(batch, predictions, all_nf)
        total_loss = loss['total_loss']
        x_loss = loss['x_loss']
        y_loss = loss['y_loss']
        z_loss = loss['z_loss']
        phase_loss = loss['phase_loss']
        derivative_loss = loss['derivative_loss']
        thermo_loss = loss['thermo_loss']
       
        #Log the loss
        self.log("val_total_loss", total_loss, prog_bar = True, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_nf_x_loss", x_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_nf_y_loss", y_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_nf_z_loss", z_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_phase_loss", phase_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_derivative_loss", derivative_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_thermo_loss", thermo_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)
        #self.log("source_flux_loss", source_flux_loss, prog_bar = False, on_step = False, on_epoch = True, sync_dist = True)

        return {'loss' : total_loss, 'output' : predictions, 'target' : batch}
 
    def test_step(self, batch, batch_idx, dataloader_idx=0): # this lets us do evaluation
        #Get predictions
        all_nf, predictions = self.shared_step(batch, batch_idx) # grabs all preds from the dataloader
        
        pred_nf, pred_phases, pred_derivatives = predictions[0], predictions[1], predictions[2] # only using pred_nf here so we can calculate pred_intensities. using predictions as a list.
        pred_intensities = self.get_pred_intensities(pred_nf) 
        predictions.append(pred_intensities) 

        # predictions passed as a list. [0]: pred_nf, [1]: pred_phases, [2]: pred_derivatives, [3]: pred_intensities
        self.organize_testing(predictions, batch, batch_idx, dataloader_idx) # fills in those lists we defined above.
        

    def on_test_end(self):         
         
        # Encoder
        train_encoder = {
            'phase_truth'     : np.concatenate(self.train_phase_truth),
            'phase_pred'      : np.concatenate(self.train_phase_pred),
            'deriv_truth'     : np.concatenate(self.train_deriv_truth),
            'deriv_pred'      : np.concatenate(self.train_deriv_pred),
            'intensity_truth' : np.concatenate(self.train_intensity_truth),
            'intensity_pred'  : np.concatenate(self.train_intensity_pred),
            }
        val_encoder = {
            'phase_truth'     : np.concatenate(self.val_phase_truth),
            'phase_pred'      : np.concatenate(self.val_phase_pred),
            'deriv_truth'     : np.concatenate(self.val_deriv_truth),
            'deriv_pred'      : np.concatenate(self.val_deriv_pred),
            'intensity_truth' : np.concatenate(self.val_intensity_truth),
            'intensity_pred'  : np.concatenate(self.val_intensity_pred),
            }

        # Resim
        train_resim = {
            'phase_pred'        : np.concatenate(self.train_phase_pred_resim),
            'phase_truth'       : np.concatenate(self.train_phase_truth_resim),
            'intensity_pred'    : np.concatenate(self.train_intensity_pred_resim),
            'intensity_truth'   : np.concatenate(self.train_intensity_truth_resim),
            'nf_2881_pred'      : np.concatenate(self.train_nf_2881_pred_resim),
            'nf_2881_truth'     : np.concatenate(self.train_nf_2881_truth_resim),
            'nf_1650_pred'      : np.concatenate(self.train_nf_1650_pred_resim),
            'nf_1650_truth'     : np.concatenate(self.train_nf_1650_truth_resim),
            'nf_1550_pred'      : np.concatenate(self.train_nf_1550_pred_resim),
            'nf_1550_truth'     : np.concatenate(self.train_nf_1550_truth_resim),
            'nf_1300_pred'      : np.concatenate(self.train_nf_1300_pred_resim),
            'nf_1300_truth'     : np.concatenate(self.train_nf_1300_truth_resim),
            'nf_1060_pred'      : np.concatenate(self.train_nf_1060_pred_resim),
            'nf_1060_truth'     : np.concatenate(self.train_nf_1060_truth_resim),
            }

        val_resim = {
            'phase_pred'        : np.concatenate(self.val_phase_pred_resim),
            'phase_truth'       : np.concatenate(self.val_phase_truth_resim),
            'intensity_pred'    : np.concatenate(self.val_intensity_pred_resim),
            'intensity_truth'   : np.concatenate(self.val_intensity_truth_resim),
            'nf_2881_pred'      : np.concatenate(self.val_nf_2881_pred_resim),
            'nf_2881_truth'     : np.concatenate(self.val_nf_2881_truth_resim),
            'nf_1650_pred'      : np.concatenate(self.val_nf_1650_pred_resim),
            'nf_1650_truth'     : np.concatenate(self.val_nf_1650_truth_resim),
            'nf_1550_pred'      : np.concatenate(self.val_nf_1550_pred_resim),
            'nf_1550_truth'     : np.concatenate(self.val_nf_1550_truth_resim),
            'nf_1300_pred'      : np.concatenate(self.val_nf_1300_pred_resim),
            'nf_1300_truth'     : np.concatenate(self.val_nf_1300_truth_resim),
            'nf_1060_pred'      : np.concatenate(self.val_nf_1060_pred_resim),
            'nf_1060_truth'     : np.concatenate(self.val_nf_1060_truth_resim),
            }

        train_recon = {
            'nf_2881_amp_diff'       : np.concatenate(self.train_nf_2881_amp_diff),
            'nf_2881_angle_diff'     : np.concatenate(self.train_nf_2881_angle_diff),
            'nf_1650_amp_diff'       : np.concatenate(self.train_nf_1650_amp_diff),
            'nf_1650_angle_diff'     : np.concatenate(self.train_nf_1650_angle_diff),
            'nf_1550_amp_diff'       : np.concatenate(self.train_nf_1550_amp_diff),
            'nf_1550_angle_diff'     : np.concatenate(self.train_nf_1550_angle_diff),
            'nf_1300_amp_diff'       : np.concatenate(self.train_nf_1300_amp_diff),
            'nf_1300_angle_diff'     : np.concatenate(self.train_nf_1300_angle_diff),
            'nf_1060_amp_diff'       : np.concatenate(self.train_nf_1060_amp_diff),
            'nf_1060_angle_diff'     : np.concatenate(self.train_nf_1060_angle_diff),

            }
        val_recon = {
            'nf_2881_amp_diff'       : np.concatenate(self.val_nf_2881_amp_diff),
            'nf_2881_angle_diff'     : np.concatenate(self.val_nf_2881_angle_diff),
            'nf_1650_amp_diff'       : np.concatenate(self.val_nf_1650_amp_diff),
            'nf_1650_angle_diff'     : np.concatenate(self.val_nf_1650_angle_diff),
            'nf_1550_amp_diff'       : np.concatenate(self.val_nf_1550_amp_diff),
            'nf_1550_angle_diff'     : np.concatenate(self.val_nf_1550_angle_diff),
            'nf_1300_amp_diff'       : np.concatenate(self.val_nf_1300_amp_diff),
            'nf_1300_angle_diff'     : np.concatenate(self.val_nf_1300_angle_diff),
            'nf_1060_amp_diff'       : np.concatenate(self.val_nf_1060_amp_diff),
            'nf_1060_angle_diff'     : np.concatenate(self.val_nf_1060_angle_diff),

            }        
        #log_results(self, results, epoch, mode, count = 5, name = None):       
        self.logger.experiment.log_results(results = train_encoder, epoch=None, count=5, mode = 'train', name='encoder')
        self.logger.experiment.log_results(results = train_resim, epoch=None, count=5, mode = 'train', name='resim')
        self.logger.experiment.log_results(results = train_recon, epoch=None, count=5, mode = 'train', name='recon')

        self.logger.experiment.log_results(results = val_encoder, epoch=None, count=5, mode = 'val', name='encoder')
        self.logger.experiment.log_results(results = val_resim, epoch=None, count=5, mode = 'val', name='resim')
        self.logger.experiment.log_results(results = val_recon, epoch=None, count=5, mode = 'val', name='recon')

  
if __name__ == "__main__":
    
    from pytorch_lightning import seed_everything
    from core import datamodule
    seed_everything(1337)
    params = yaml.load(open('../config.yaml', 'r'), Loader = yaml.FullLoader)
    pm = parameter_manager.ParameterManager(params = params)

    dm = datamodule.select_data(pm.params_datamodule)
    dm.prepare_data()
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    #model = CAI_Model(pm.params_model, pm.params_propagator)
    #model = Encoder(pm.params_model)
    
    model = SurrogateModel()
    embed() 
