from IPython import embed
import pickle
import sys
import os
import torch
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import PeakSignalNoiseRatio
sys.path.append(os.path.dirname(os.getcwd()))
from utils import parameter_manager
sys.path.append("../core")
from core import datamodule, model, custom_logger, curvature
from model import SurrogateModel
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm
import scipy.stats as stats

fontsize = 8
font = FontProperties()

psnr = PeakSignalNoiseRatio()

torch.multiprocessing.set_sharing_strategy('file_system')

pl.seed_everything(1337)
path_results = "/develop/results/spie_journal_2023"
path_resims = "/develop/resims/resim_outputs"
ckpt_path = f"/develop/model_checkpoints"

colors = ['darkgreen','purple','#4e88d9'] 

#### Gather loss info #####
###########################

def gather_loss(folder_path):
    excess = os.path.join(path_results, "model_cai_")
    file_path = os.path.join(folder_path, "params.yaml")

    if os.path.isfile(file_path):

        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            key_dict = {}    
            
            alpha = key_dict['alpha'] = yaml_content['mcl_params']['alpha']
            #beta = key_dict['beta'] = yaml_content['mcl_params']['beta']
            delta = key_dict['delta'] = yaml_content['mcl_params']['delta']
            gamma = key_dict['gamma'] = yaml_content['mcl_params']['gamma']
            lr = key_dict['lr'] = yaml_content['learning_rate']
            key_dict['title'] = folder_path.replace(excess, "")

            loss_file = os.path.join(folder_path, "loss.csv")
            loss = load_loss(loss_file)
            if type(loss) == bool and loss == False:
                print("Failed to load loss")
                pass
            else:
                key_dict['loss'] = loss
                return key_dict
                
    else:
        print(f"{file_path} does not exist")


def gather_all_loss(path_results, type, backbone):
    dict_list = []
    num_folders = 0
    num_with_loss = 0

    processed_folders = set()
    
    for folder in os.listdir(path_results):
        if type in folder:
            folder_path = os.path.join(path_results, folder)
            #print(f'folder_path = {folder_path}')
            if os.path.isdir(folder_path):
            
                #print(f'processed folders={processed_folders}')
                num_folders+= 1
                params_yaml_processed = False
                for filename in os.listdir(folder_path):
                
                #file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(os.path.join(folder_path,"params.yaml")):
                        params_yaml_processed = True
                        temp = os.path.join(folder_path,"params.yaml")
                        #key_dict = em.gather_loss(os.path.join(folder_path,"params.yaml"))
                        #print(folder_path)
                        key_dict = gather_loss(folder_path)
                        #print(key_dict)
                        #if key_dict:
                        if folder_path not in processed_folders:
                            dict_list.append(key_dict)
                            processed_folders.add(folder_path)
                            num_with_loss += 1
                    
    print(f"num folders found: {num_folders}")
    print(f"num with loss: {num_with_loss}")
    return dict_list

def is_csv_empty(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return True
    except pd.errors.EmptyDataError:
        return True

    return False

def load_loss(path):
    if is_csv_empty(path):
        print("Empty CSV file")
        return False
    else:
        return pd.read_csv(path)

# ['val_total_loss', 'val_nf_x_loss', 'val_nf_y_loss', 'val_nf_z_loss',
#        'val_phase_loss', 'val_derivative_loss', 'val_thermo_loss', 'epoch',
#        'train_total_loss', 'train_nf_x_loss', 'train_nf_y_loss',
#        'train_nf_z_loss', 'train_phase_loss', 'train_derivative_loss',
#        'train_thermo_loss']

def plot_loss(loss_0, loss_1, loss_2, loss_3, title, rect=[0,0,0.98,1], save_fig=False):
    
    plt.style.use("ggplot")
    
    fig, ax = plt.subplots(1, 4, figsize = (11, 3))
    
    fig.suptitle("Multi-stage loss, " + title)
        
    lterms = ['Phase Loss', 'Curvature Loss', 'Thermo Loss', 'Near Field Loss'] # near field: alpha, phase: gamma, derivs: delta
    labels = ["Train", "Valid"]
    linestyles = ['solid', 'dotted', 'dashdot']
    colors = ['darkgreen','purple','#4e88d9'] 
    # train dataset

    # phase
    x_vals = loss_0['epoch']
    y_vals = loss_0['train_phase_loss']
    x_vals = x_vals[x_vals.index % 2 != 0]
    y_vals = y_vals[y_vals.index % 2 != 0]

    ax[0].plot(x_vals, y_vals, label=labels[0], c=colors[0])
    ax[0].legend()

    # der
    x_vals = loss_1['epoch']
    y_vals = loss_1['train_derivative_loss']
    x_vals = x_vals[x_vals.index % 2 != 0]
    y_vals = y_vals[y_vals.index % 2 != 0]
    
    ax[1].plot(x_vals, y_vals, label=labels[0], c=colors[0])
    ax[1].legend()

    # thermo
    x_vals = loss_2['epoch']
    y_vals = loss_2['train_thermo_loss']
    x_vals = x_vals[x_vals.index % 2 != 0]
    y_vals = y_vals[y_vals.index % 2 != 0]

    ax[2].plot(x_vals, y_vals, label=labels[0], c=colors[0])
    ax[2].legend()
    
    # nf
    x_vals = loss_3['epoch']
    y_vals = loss_3['train_nf_x_loss']
    x_vals = x_vals[x_vals.index % 2 != 0]
    y_vals = y_vals[y_vals.index % 2 != 0]
    
    ax[3].plot(x_vals, y_vals, label=labels[0] + " x", c=colors[0], linestyle=linestyles[0])

    x_vals = loss_3['epoch']
    y_vals = loss_3['train_nf_y_loss']
    x_vals = x_vals[x_vals.index % 2 != 0]
    y_vals = y_vals[y_vals.index % 2 != 0]
    
    ax[3].plot(x_vals, y_vals, label=labels[0] + " y", c=colors[0], linestyle=linestyles[1])

    x_vals = loss_3['epoch']
    y_vals = loss_3['train_nf_z_loss']
    x_vals = x_vals[x_vals.index % 2 != 0]
    y_vals = y_vals[y_vals.index % 2 != 0]
    
    ax[3].plot(x_vals, y_vals, label=labels[0] + " z", c=colors[0], linestyle=linestyles[2])
    ax[3].legend()
               
    # Valid dataset

    # phase
    x_vals = loss_0['epoch']
    y_vals = loss_0['val_phase_loss']
    x_vals = x_vals[x_vals.index % 2 == 0]
    y_vals = y_vals[y_vals.index % 2 == 0]
    
    ax[0].plot(x_vals, y_vals, label=labels[1], c=colors[1])
    ax[0].legend()

    # curv
    x_vals = loss_1['epoch']
    y_vals = loss_1['val_derivative_loss']
    x_vals = x_vals[x_vals.index % 2 == 0]
    y_vals = y_vals[y_vals.index % 2 == 0]
    
    ax[1].plot(x_vals, y_vals, label=labels[1], c=colors[1])
    ax[1].legend()

    # thermo
    x_vals = loss_2['epoch']
    y_vals = loss_2['val_thermo_loss']
    x_vals = x_vals[x_vals.index % 2 == 0]
    y_vals = y_vals[y_vals.index % 2 == 0]

    ax[2].plot(x_vals, y_vals, label=labels[1], c=colors[1])
    ax[2].legend()
    
    x_vals = loss_3['epoch']
    y_vals = loss_3['val_nf_x_loss']
    x_vals = x_vals[x_vals.index % 2 == 0]
    y_vals = y_vals[y_vals.index % 2 == 0]

    ax[3].plot(x_vals, y_vals, label=labels[1] + " x", c=colors[1], linestyle=linestyles[0])

    x_vals = loss_3['epoch']
    y_vals = loss_3['val_nf_y_loss']
    x_vals = x_vals[x_vals.index % 2 == 0]
    y_vals = y_vals[y_vals.index % 2 == 0]
    
    ax[3].plot(x_vals, y_vals, label=labels[1] + " y", c=colors[1], linestyle=linestyles[1])

    x_vals = loss_3['epoch']
    y_vals = loss_3['val_nf_z_loss']
    x_vals = x_vals[x_vals.index % 2 == 0]
    y_vals = y_vals[y_vals.index % 2 == 0]
    
    ax[3].plot(x_vals, y_vals, label=labels[1] + " z", c=colors[1], linestyle=linestyles[2])
    ax[3].legend(loc='upper left', bbox_to_anchor=(1,1))

    lims = [6, 9, 0.3, 5]
    for a, lterm, lim in zip(ax, lterms, lims):
        a.set_xlabel("Epoch")
        a.set_ylabel("Loss")
        a.set_title(lterm)
        a.set_ylim([0,lim])
    
    fig.tight_layout(rect=rect)

    if save_fig == True:
        fig.savefig(f"images/loss/{title}.pdf")

#### Get model results ####
###########################

def load_model(folder_name, batch_size=1, device='cpu'):

    params_path = os.path.join(path_results, f"{folder_name}/params.yaml")

    new_ckpt_path = os.path.join(ckpt_path,folder_name)
    ckpt = os.listdir(new_ckpt_path)
    ckpt.sort()
    ckpt = ckpt[0]

    temp = torch.load(os.path.join(new_ckpt_path, ckpt))
    state_dict = temp['state_dict']
    params = yaml.load(open(params_path,"r"), Loader = yaml.FullLoader)
    params['path_resims'] = None
    params['freeze_encoder'] = False
    params['batch_size'] = batch_size
    pm = parameter_manager.Parameter_Manager(params=params)

    model = CAI_Model(pm.params_model, pm.params_propagator)
    model.load_state_dict(state_dict)
    if device == "cpu":
        return model
    elif(device == "gpu"):
        return model.cuda()
    else:
        print(f"eval_methods.py, load_model() unsupported device. try again, dumb head.")
        exit()

def load_data(batch_size, params_path):
    params = yaml.load(open(params_path, "r"), Loader=yaml.FullLoader)
    params['path_resims'] = None
    params['freeze_encoder'] = False
    pm = parameter_manager.Parameter_Manager(params=params)
    
    pm.batch_size = batch_size
    pm.collect_params()
    
    data = datamodule.select_data(pm.params_datamodule)
    data.prepare_data()
    data.setup(stage='fit')
    data_loader_train = data.train_dataloader()
    data_loader_valid = data.val_dataloader()
    
    return data_loader_train, data_loader_valid
    
def evaluate(model, data_loader):

    pred_near_field_list, pred_far_field_list, pred_phase_list, pred_derivatives_list = [], [], [], []
    truth_near_field_list, truth_far_field_list, truth_phase_list, truth_derivatives_list = [], [], [], []        
    eval_results = {}
    
    for i, batch in enumerate(tqdm(data_loader, desc="eval loop")):
        
        pred_near_field, pred_far_field, pred_phase, pred_derivatives = model.shared_step(batch, i)
        truth_near_field, truth_far_field, radii, truth_phase, truth_derivatives = batch
        pred_near_field_list.append(pred_near_field.cpu().detach().numpy())
        pred_far_field_list.append(pred_far_field.cpu().detach().numpy())
        pred_phase_list.append(pred_phase.cpu().detach().numpy())
        pred_derivatives_list.append(pred_derivatives.cpu().detach().numpy())
        truth_near_field_list.append(truth_near_field.cpu().detach().numpy())
        truth_far_field_list.append(truth_far_field.cpu().detach().numpy())
        truth_phase_list.append(truth_phase.cpu().detach().numpy())
        truth_derivatives_list.append(truth_derivatives.cpu().detach().numpy())
    
    eval_results['pred_nf'] = np.asarray(pred_near_field_list).squeeze()
    eval_results['pred_ff'] = np.asarray(pred_far_field_list).squeeze()
    eval_results['pred_phase'] = np.asarray(pred_phase_list).squeeze()
    eval_results['pred_der'] = np.asarray(pred_derivatives_list).squeeze()
    eval_results['truth_nf'] = np.asarray(truth_near_field_list).squeeze()
    eval_results['truth_ff'] = np.asarray(truth_far_field_list).squeeze()
    eval_results['truth_phase'] = np.asarray(truth_phase_list).squeeze()
    eval_results['truth_der'] = np.asarray(truth_derivatives_list).squeeze()
    return eval_results

def get_results(exp_name,stage,resim=False,resim_index=None,folder_name=None):

    if folder_name is None:
        folder_name = exp_name + "_" + str(stage)

    path_results = "/develop/results/spie_journal_2023"
    loss_file = os.path.join(path_results, folder_name, "loss.csv")
    loss = load_loss(loss_file)

    encoder_train = pickle.load(open(os.path.join(path_results, folder_name, "train_info", "encoder.pkl"), "rb"))
    recon_train = pickle.load(open(os.path.join(path_results, folder_name, "train_info", "recon.pkl"), "rb"))
    
    encoder_valid =pickle.load(open(os.path.join(path_results, folder_name, "valid_info", "encoder.pkl"), "rb"))
    recon_valid = pickle.load(open(os.path.join(path_results, folder_name, "valid_info", "recon.pkl"), "rb"))
    
    # outputs from the model
    resim_train_eval = pickle.load(open(os.path.join(path_results, folder_name, "train_info", "resim.pkl"), "rb"))
    resim_valid_eval = pickle.load(open(os.path.join(path_results, folder_name, "valid_info", "resim.pkl"), "rb"))

    # get resim results
    if resim == True:

        try:
            resim_train_results = pickle.load(open(os.path.join(path_results, folder_name, "train_info", f"sample_{resim_index}_preprocessed.pkl"), "rb"))
        except FileNotFoundError:
            resim_train_results = None
            print("no file for resim_train")
        try:
            resim_valid_results = pickle.load(open(os.path.join(path_results, folder_name, "valid_info", f"sample_{resim_index}_preprocessed.pkl"), "rb"))
        except:
            resim_valid_results = None
            print("no file for resim_valid")
        return loss, encoder_train, recon_train, encoder_valid, recon_valid, resim_train_eval, resim_valid_eval, resim_train_results, resim_valid_results
        
    elif resim == False:
        return loss, encoder_train, recon_train, encoder_valid, recon_valid, resim_train_eval, resim_valid_eval

##########################################
########## encoder eval ##################

def get_r_squared(x, y):
    coeff, p_val = stats.pearsonr(x, y)
    r_squared = coeff **2

    return r_squared

def plot_scatter(ax, truth, pred, title, x_label, y_label):

    scatter_color = "mediumseagreen"
    line_color = "black"
    linewidth=3
    an_fontsize = 12.5
    bbox = dict(boxstyle='round', facecolor='white', edgecolor='black')

    r_sq = get_r_squared(truth, pred)
    ax.scatter(truth, pred, c=colors[1])
    ax.plot([-max(truth), max(truth)], [-max(truth), max(truth)], c=scatter_color, linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.annotate(f"$R^2$: {r_sq:.2f}", (-max(truth), max(truth)), fontsize=an_fontsize, bbox=bbox)
    ax.grid(color='black')
    ax.set_facecolor('white')

def regression_plots(title, phase_train_truth, phase_train_pred, phase_valid_truth, phase_valid_pred, der_train_truth, der_train_pred, der_valid_truth, der_valid_pred, save_fig=False):
    
    x_values = np.linspace(-np.pi, np.pi, 100)
    
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle(f"{title}")
    
    plot_scatter(ax[0, 0], phase_train_truth, phase_train_pred, "Train Dataset", "Truth Phases", "Pred Phases")
    plot_scatter(ax[0, 1], phase_valid_truth, phase_valid_pred, "Valid Dataset", "Truth Phases", "Pred Phases")
    plot_scatter(ax[1, 0], der_train_truth, der_train_pred, "Train Dataset", "Truth Derivatives", "Pred Derivatives")
    plot_scatter(ax[1, 1], der_valid_truth, der_valid_pred, "Valid Dataset", "Truth Derivatives", "Pred Derivatives")
    
    fig.set_facecolor('white')
    fig.tight_layout()

    if(save_fig==True):
        fig.savefig("other_plots/regression_plots.pdf")

##########################################
#####  Decoder eval #######################

# def set_box_color(bp, color, linestyle, linewidth = 1.5):
    
#     plt.setp(bp['boxes'], color=color, linewidth=linewidth,  linestyle=linestyle, dashes=(1, 0.2))
#     plt.setp(bp['whiskers'], color=color, linewidth=linewidth, linestyle=linestyle, dashes=(1, 0.2))
#     plt.setp(bp['caps'], color=color, linewidth=linewidth)
#     plt.setp(bp['medians'], color=color, linewidth=linewidth)

def set_violin_color(vp, color, linewidth=2, linestyle='dotted'):
    for pc in vp['bodies']:    
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_linewidth(linewidth)

    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        dashes = (1, 0.2)
        dashes = [int(dash * 10) for dash in dashes]
        part = vp[partname]
        part.set_edgecolor(color)
        if 'mean' in partname:
            part.set_linestyle('dotted')
            part.set_linewidth(linewidth)
            #part.set_dashes(np.array(dashes))

def violin_plots(nf_amp_diff, nf_angle_diff, dataset, savefig=False):

    plt.style.use("ggplot")
    
    num_models = 1       # (integer) This sets how many boxes next to each other per index
    buffer = 0.5         # (float)   This sets the spacing between boxes for the same index
    num_values =  2      # (integer) This sets the number of indices (9 for phase, 6 for curvature)
    position_offset = 1  # (integer) This sets the offset of the indices - increase if neighboring 'groups' are too close
    fontsize = 14

    pos = np.array(range(num_values))*(num_models+position_offset) 
    fig,ax = plt.subplots(1,1,figsize=(8,5))

    nf_amp_diff = np.mean(nf_amp_diff, axis=(1,2))
    nf_angle_diff = np.mean(nf_angle_diff, axis=(1,2))
    #bp0 = ax.boxplot((nf_amp_diff, nf_angle_diff), sym='', positions = pos - buffer, widths=0.6)
    vp = ax.violinplot((nf_amp_diff, nf_angle_diff), positions = pos, widths=0.6, showmeans=True, showmedians=True, showextrema=True, 
                       points=len(nf_amp_diff))

    vp['bodies'][0].set_facecolor(colors[0])
    vp['bodies'][0].set_edgecolor(colors[0])
    vp['bodies'][1].set_facecolor(colors[1])
    vp['bodies'][1].set_edgecolor(colors[1])
        #pc.set_linewidth(linewidth)

    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        dashes = (1, 0.2)
        dashes = [int(dash * 10) for dash in dashes]
        part = vp[partname]
        part.set_edgecolor('black')
        if 'mean' in partname:
            part.set_linestyle('dotted')
            part.set_linewidth(1)
    
    ax.tick_params(axis='both', labelsize=fontsize)
    
    ax.set_xticks(range(0, 2*(num_models+position_offset), (num_models+position_offset)))
    ax.set_xticklabels(['Amplitude', 'Phase'])
    ax.set_xlim(-(buffer*2), 2*((num_models)+position_offset)-(buffer*2))
    #ax.set_ylim(-0.1,4)
    ax.grid(axis='x', which='both', linewidth=0)  # Set linewidth to 0 to hide the grid lines
    ax.grid(axis='y', which='both', linewidth=1, linestyle='--', color='black', alpha=0.7)  # Set linewidth to 0 to hide the grid lines

    ax.set_ylabel("Average Absolute Difference")
    
    #Gotta do this to get the legend
    #Plots empty lines with the correct color and linestyle to match. We will use these in the legend. 
    # ln0, = ax.plot([], c=colors[0], label='Amplitude', linestyle = 'solid')
    # ln1, = ax.plot([], c=colors[1], label='Phase', linestyle = 'solid')
    # leg = ax.legend(handles = [ln0, ln1] , loc='upper left', frameon = True)

    # leg.get_frame().set_edgecolor('gray')

    ax.set_title("Reconstruction : {}".format(dataset))
    
    plt.tight_layout()
    if(savefig==True):
        fig.savefig("other_plots/recon_boxplot_{}.pdf".format(dataset))
    plt.show()

# def box_plots(nf_amp_diff, nf_angle_diff, ff_amp_diff, ff_angle_diff, dataset):
    
#     num_models = 2       # (integer) This sets how many boxes next to each other per index
#     buffer = 0.5         # (float)   This sets the spacing between boxes for the same index
#     num_values =  2      # (integer) This sets the number of indices (9 for phase, 6 for curvature)
#     position_offset = 1  # (integer) This sets the offset of the indices - increase if neighboring 'groups' are too close
#     fontsize = 14

#     pos = np.array(range(num_values))*(num_models+position_offset) 
#     fig,ax = plt.subplots(1,1,figsize=(8,5))

#     nf_amp_diff = np.mean(nf_amp_diff, axis=(1,2))
#     nf_angle_diff = np.mean(nf_angle_diff, axis=(1,2))
#     bp0 = ax.boxplot((nf_amp_diff, nf_angle_diff), sym='', positions = pos - buffer, widths=0.6)


#     ff_amp_diff = np.mean(ff_amp_diff, axis=(1,2))
#     ff_angle_diff = np.mean(ff_angle_diff, axis=(1,2))
#     bp1 = ax.boxplot((ff_amp_diff, ff_angle_diff), sym='', positions = pos + buffer, widths=0.6)

#     set_box_color(bp0, colors[0], 'solid')
#     set_box_color(bp1, colors[1], 'solid')

#     # ax.set_ylabel("Phase Difference", fontsize=fontsize)
#     # ax.set_xlabel("Phase Index", fontsize=fontsize)
    
#     ax.tick_params(axis='both', labelsize=fontsize)
    
#     ax.set_xticks(range(0, 2*(num_models+position_offset), (num_models+position_offset)))
#     ax.set_xticklabels(['Near Field', 'Far Field'])
#     ax.set_xlim(-(buffer*2), 2*((num_models)+position_offset)-(buffer*2))
#     ax.set_ylim(-0.1,4)
#     ax.grid(axis='x', which='both', linewidth=0)  # Set linewidth to 0 to hide the grid lines
#     ax.grid(axis='y', which='both', linewidth=1, linestyle='--', color='black', alpha=0.7)  # Set linewidth to 0 to hide the grid lines

#     ax.set_ylabel("Average absolute difference")
    
#     #Gotta do this to get the legend
#     #Plots empty lines with the correct color and linestyle to match. We will use these in the legend. 
#     ln0, = ax.plot([], c=colors[0], label='Amplitude', linestyle = 'solid')
#     ln1, = ax.plot([], c=colors[1], label='Phase', linestyle = 'solid')
#     leg = ax.legend(handles = [ln0, ln1] , loc='upper left', frameon = True)

#     leg.get_frame().set_edgecolor('gray')

#     ax.set_title("Reconstruction : {}".format(dataset))
    
#     plt.tight_layout()
#     fig.savefig("recon_boxplot_{}.pdf".format(dataset))
#     plt.show()
    
###########################################
############ Resim stuff ##################

def dump_results_for_resim(folder_name, train_results, valid_results, target=5):
    destination_path = "/develop/resims/radii_predictions"
    results = {}
    temp = {}

    for k in train_results:
        temp[k] = train_results[k][:target]
    results['train_results'] = temp
    temp = {}
    for k in valid_results:
        temp[k] = valid_results[k][:target]
    results['valid_results'] = temp

    pickle.dump(results, open(os.path.join(destination_path, f"{folder_name}.pkl"), "wb"))

def calculate_matrix(original, recon, resim, similarity = True):

    if similarity is True:
        resim_vs_original = psnr(resim, original)
        resim_vs_recon = psnr(resim, recon)
        resim_vs_resim = psnr(resim, resim)
        
        original_vs_original = psnr(original, original)
        original_vs_recon = psnr(original, recon)
        original_vs_resim = psnr(original, resim)
    
        recon_vs_original = psnr(recon, original)
        recon_vs_recon = psnr(recon, recon)
        recon_vs_resim = psnr(recon, resim)
    else:
        resim_vs_original = 1 / (1 + psnr(resim, original))
        resim_vs_recon = 1 / (1 + psnr(resim, recon))
        resim_vs_resim = 1 / (1 + psnr(resim, resim))

        original_vs_original = 1 / (1 + psnr(original, original))
        original_vs_recon = 1 / (1 + psnr(original, recon))
        original_vs_resim = 1 / (1 + psnr(original, resim))
    
        recon_vs_original = 1 / (1 + psnr(recon, original))
        recon_vs_recon = 1 / (1 + psnr(recon, recon))
        recon_vs_resim = 1 / (1 + psnr(recon, resim))

    row0 = np.asarray([original_vs_original,original_vs_recon,original_vs_resim])
    row1 = np.asarray([recon_vs_original,recon_vs_recon,recon_vs_resim])
    row2 = np.asarray([resim_vs_original,resim_vs_recon,resim_vs_resim])

    conf_matrix = np.asarray([row0, row1, row2])

    conf_matrix[conf_matrix==np.inf] = 0
    
    return conf_matrix

def build_custom_conf_matrices(original, recon, resim, similarity = True, savefig = False):

    resim = np.asarray(resim)
    resim_amp =np.abs(resim)
    resim_phase = np.angle(resim)

    resim = np.stack([resim_amp, resim_phase], axis=1)
    
    original = torch.from_numpy(original)
    recon = torch.from_numpy(recon)
    resim = torch.from_numpy(resim)
    
    original_amplitude = original[:,0,:,:]
    recon_amplitude = recon[:,0,:,:]
    resim_amplitude = resim[:,0,:,:]

    original_phase = original[:,1,:,:]
    recon_phase = recon[:,1,:,:]
    resim_phase = resim[:,1,:,:]

    amplitude_conf_matrix = calculate_matrix(original_amplitude, recon_amplitude, resim_amplitude, similarity)
    phase_conf_matrix = calculate_matrix(original_phase, recon_phase, resim_phase, similarity)

    return amplitude_conf_matrix, phase_conf_matrix

def plot_custom_confusion_matrix(amplitude_matrix, phase_matrix, similarity, savefig=False):
    
    fig,ax = plt.subplots(1,2,figsize=(8,5))

    plt.suptitle("PSNR",fontsize=14) if similarity is True else plt.suptitle("Dissimilarity",fontsize=14)
    
    cmap = 'Blues'
    ax[0].imshow(amplitude_matrix, cmap=cmap, alpha=1)
    ax[0].set_title('Amplitude', fontsize=10)
    
    ax[1].imshow(phase_matrix, cmap=cmap, alpha=1)
    ax[1].set_title('Phase', fontsize=10)
    
    ax[0].set_xticks([0,1,2])
    ax[0].set_yticks([0,1,2])
    
    ax[0].set_xticklabels(['Original', 'Recon', 'Resim'], fontsize=10)
    ax[0].set_yticklabels(['Original', 'Recon', 'Resim'], fontsize=10)
    #ax[0].yaxis.tick_right()
    ax[0].yaxis.set_tick_params(rotation=90)
    
    ax[1].set_xticks([0,1,2])
    ax[1].set_yticks([0,1,2])
    
    ax[1].set_xticklabels(['Original', 'Recon', 'Resim'], fontsize=10)
    ax[1].set_yticklabels(['Original', 'Recon', 'Resim'], fontsize=10)
    ax[1].yaxis.set_tick_params(rotation=90)
    #ax[1].yaxis.tick_right()
    
    for i in ax:
        i.grid(False)

    cmap_blues = cm.get_cmap(cmap)
    dark_blue = cmap_blues(0.95)
    
    for i in range(amplitude_matrix.shape[0]):
        for j in range(amplitude_matrix.shape[1]):
            value = amplitude_matrix[i, j]
            color = 'white' if value != 0 else dark_blue
            ax[0].text(x=j, y=i,s=round(amplitude_matrix[i, j],2), va='center',ha='center', size='xx-large', color=color)
    
    
    for i in range(phase_matrix.shape[0]):
        for j in range(phase_matrix.shape[1]):
            value = amplitude_matrix[i, j]
            color = 'white' if value != 0 else dark_blue
            ax[1].text(x=j, y=i,s=round(phase_matrix[i, j],2), va='center',ha='center', size='xx-large', color=color)

    if savefig == True:
        title = 'PSNR' if similarity else 'Dissimilarity'
        fig.savefig(f"other_plots/{title}.pdf")
        
    plt.tight_layout()

######################################
########## resim eval stuff ##########

def get_nf_resim(folder_name, target):
    
    folder = os.path.join(folder_name, target + "_info")
    nf_resim = []
    # for folder in os.listdir(path_resims):
    #     if folder_name in folder:
    #         if target in folder:
    #             resim = pickle.load(open(os.path.join(path_resims,folder),"rb"))
    #             nf_resim.append(resim['near_fields']['grating_ey'])
    return nf_resim

def plot_dft_fields(truth, recon, resim, idx=0, savefig=False, id=None): # idx refers to the sample number in the batch, id is identifier if we save the image
    
    cmap = 'turbo'
    fig, ax = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(5,8), gridspec_kw={'width_ratios': [0.03, 1, 1]})
    fig.suptitle(f"Electric Fields, y component, {id} dataset")

    mag_truth = truth[idx][0]
    angle_truth = truth[idx][1]
    mag_recon = recon[idx][1][0]
    angle_recon = recon[idx][1][1]
    mag_resim = resim[0][1][0]
    angle_resim = resim[0][1][1]

    ax[0][1].imshow(mag_truth**2, cmap = cmap, vmin = 0)    
    ax[0][2].imshow(angle_truth, cmap = cmap, vmin = -torch.pi, vmax = torch.pi)
    
    ax[1][1].imshow(mag_recon**2, cmap = cmap, vmin = 0)
    ax[1][2].imshow(angle_recon, cmap = cmap, vmin = -torch.pi, vmax = torch.pi)
    
    ax[2][1].imshow(mag_resim**2, cmap = cmap, vmin = 0)
    ax[2][2].imshow(angle_resim, cmap = cmap, vmin = -torch.pi, vmax = torch.pi)

    column_titles = ['Intensity', 'Phase']
    row_titles = ['Original/Truth', 'Recon', 'Resim']  # these row titles won't appear with axes off

    for j in range(2):
        ax[0][j+1].set_title(column_titles[j], fontsize=10)
    
    for i in range(3):
        ax[i][0].text(-0.5, 0.5, row_titles[i], transform = ax[i][0].transAxes, rotation=90, ha='center', va='center', fontsize=10)
        ax[i][0].axis("off")
        
    for i in range(3):
        for j in range(1, 3):
            ax[i][j].grid(False)
            ax[i][j].axis("off")
        
    fig.tight_layout()
    
    if savefig == True:
        #flag = 'batch' if batch else f'single_sample_idx_{idx}'
        fig.savefig(f'other_plots/{id}.pdf')


if __name__=="__main__":
    path_results = "/develop/results"
    folder_name = "model_baseline" # this is where we send loss.csv, params.yaml and the folders valid_info, train_info
    
    folder_path = os.path.join(path_results, folder_name)
    single_loss = gather_loss(folder_path)
    #all_loss = em.gather_all_loss(path_results, backbone="resnet18")
    from IPython import embed;embed()
    l_string = r'($\alpha$' + " " + r'$\gamma$' + " " + r'$\delta$)'
    title = (l_string + f": ({single_loss['alpha']} {single_loss['delta']} {single_loss['gamma']}) " + "\n" 
                    + single_loss['title'].replace(path_results,"").replace("/","") + " lr=" + str(single_loss['lr']))

    plot_loss(single_loss['loss'], title, save_fig=False)
