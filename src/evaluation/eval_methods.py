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
from core import datamodule, model, custom_logger, curvature, propagator
from model import CAI_Model
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
path_results = "/develop/results/"
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
            beta = key_dict['beta'] = yaml_content['mcl_params']['beta']
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

def plot_loss(loss, title, min, max, save_fig=False):
    
    plt.style.use("ggplot")
    
    fig, ax = plt.subplots(2, int((len(loss.keys()) - 1)/2), figsize = (12, 4.5))
        
    lterms = ['Total Loss', 'Near Field Loss', 'Far Field Loss', 'Phase Loss', 'Curvature Loss']
    
    for i, name in enumerate(loss.keys()):
        
        if(name == "epoch"):
            continue

        if(name.startswith('val')):
            #fig.suptitle(f"\nLoss Analysis {title}: " + "Valid Dataset", fontsize=fontsize, fontproperties=font)
            
            x_vals = loss["epoch"]
            x_vals = x_vals[x_vals.index % 2 == 0]
            y_vals = loss[name]
            y_vals = y_vals[y_vals.index % 2 == 0]
            
            ax[1][i].plot(x_vals, y_vals, color = colors[0], label=title)
            ax[1][i].set_ylabel("Loss", fontsize = fontsize-1)
            ax[1][i].set_xlabel("Epoch", fontsize = fontsize-1)
            ax[1][i].set_title(name,fontsize=fontsize+1)
            ax[1][i].tick_params(axis="x", labelsize=6)
            ax[1][i].tick_params(axis="y", labelsize=6)
            #ax[1][i].legend(loc='upper right', fontsize=fontsize)
            ax[1][i].set_ylim([min[i],max[i]])
            
        else:
            #fig.suptitle(f"\nLoss Analysis {title}: " + "Train Dataset", fontsize=fontsize, fontproperties=font)

            x_vals = loss["epoch"]
            x_vals = x_vals[x_vals.index % 2 != 0]
            y_vals = loss[name]
            y_vals = y_vals[y_vals.index % 2 != 0]
            
            ax[0][i-6].plot(x_vals, y_vals, color = colors[1], label=title)
            ax[0][i-6].set_ylabel("Loss", fontsize = fontsize-1)
            ax[0][i-6].set_xlabel("Epoch", fontsize = fontsize-1)
            ax[0][i-6].set_title(name,fontsize=fontsize+1)
            ax[0][i-6].tick_params(axis="x", labelsize=6)
            ax[0][i-6].tick_params(axis="y", labelsize=6)
            ax[0][i-6].set_ylim([0,100])
            #ax[0][i-6].legend(loc='upper right', fontsize=fontsize)
            ax[0][i-6].set_ylim([min[i-6],max[i-6]])
            
    fig.suptitle(title)
    fig.tight_layout()

    if save_fig == True:
        l_string = r'($\alpha$' + " " + r'$\beta$' + " " + r'$\gamma$' + " " + r'$\delta$)'
        temp = title.replace(l_string, "").replace("\n", "").replace(": ", "").replace(" ", "_").replace("=", "").replace("(", "").replace(")","").replace(".", "-")
        new_title = "a_b_d_g_" + temp
        
        if(fig.savefig(f'loss_plots/{new_title}.pdf')):
            print("fig saved")

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

def get_results_outside_training(folder_name, batch_size=1):

    params_path = os.path.join(path_results, f"{folder_name}/params.yaml")
    batch_size = 1

    model = load_model(folder_name, batch_size)
    data_loader_train, data_loader_valid = load_data(batch_size, params_path)
    
    train_results = evaluate(model, data_loader_train)
    valid_results = evaluate(model, data_loader_valid)

    return train_results, valid_results

def get_results(folder_name,target):

    train_path = os.path.join(path_results, folder_name, "train_info")
    valid_path = os.path.join(path_results, folder_name, "valid_info")

    train_results = pickle.load(open(os.path.join(train_path,target),"rb"))
    valid_results = pickle.load(open(os.path.join(valid_path,target),"rb"))
    
    return train_results, valid_results

##########################################
########## encoder eval ##################

def get_r_squared(x, y):
    coeff, p_val = stats.pearsonr(x, y)
    r_squared = coeff **2

    return r_squared

def get_regression_plots(train_results, valid_results, title, save_fig=False):

    plt.style.use('ggplot')
    
    linewidth=3
    scatter_color = "mediumseagreen"
    #line_color = "#80004C"
    line_color = "black"
    an_fontsize = 12.5
    bbox = dict(boxstyle='round', facecolor='white',   edgecolor='black')
    
    train_phase_truth = train_results['phase_truth'].flatten()
    train_phase_pred = train_results['phase_pred'].flatten()
    train_der_truth = train_results['deriv_truth'].flatten()
    train_der_pred = train_results['deriv_pred'].flatten()
    
    valid_phase_truth = valid_results['phase_truth'].flatten()
    valid_phase_pred = valid_results['phase_pred'].flatten()
    valid_der_truth = train_results['deriv_truth'].flatten()
    valid_der_pred = train_results['deriv_pred'].flatten()

    r_sq = get_r_squared(train_phase_truth, train_phase_pred)
    x = np.linspace(-np.pi, np.pi, 100)
    y = x
    
    fig, ax = plt.subplots(2, 2, figsize = (9,6))
    fig.suptitle("Encoder Evaluation")
    
    ax[0, 0].scatter(train_phase_truth, train_phase_pred, c=colors[1])
    ax[0, 0].plot(x,y, c=scatter_color, linewidth=linewidth)
    ax[0, 0].set_title("Train Dataset")
    ax[0, 0].set_xlabel("Truth Phases")
    ax[0, 0].set_ylabel("Pred Phases")
    ax[0, 0].annotate(f"$R^2$: {r_sq: .2f}", (-3, 2), fontsize=an_fontsize, bbox= bbox)
    ax[0, 0].grid(color='black')
    ax[0, 0].set_facecolor('white')
    
    r_sq = get_r_squared(valid_phase_truth, valid_phase_pred)

    ax[0, 1].scatter(valid_phase_truth, valid_phase_pred, c= colors[1])
    ax[0, 1].plot(x,y, c=scatter_color,linewidth=linewidth)
    ax[0, 1].set_title("Valid Dataset")
    ax[0, 1].set_xlabel("Truth Phases")
    ax[0, 1].set_ylabel("Pred Phases")
    ax[0, 1].annotate(f"$R^2$: {r_sq: .2f}", (-2.1, 2.5),zorder=1, fontsize=an_fontsize, bbox= bbox)
    ax[0, 1].grid(color='black')
    ax[0, 1].set_facecolor('white')
    
    r_sq = get_r_squared(train_der_truth, train_der_pred)
    x = np.linspace(-12, 12, 100)
    y = x

    ax[1, 0].scatter(train_der_truth, train_der_pred, c = colors[1])
    ax[1, 0].plot(x,y, c = scatter_color, linewidth=linewidth)
    ax[1, 0].set_title("Train Dataset")
    ax[1, 0].set_xlabel("Truth Derivatives")
    ax[1, 0].set_ylabel("Pred Derivatives")
    ax[1, 0].annotate(f"$R^2$: {r_sq: .2f}", (-10, 5), fontsize=an_fontsize, bbox= bbox)
    ax[1, 0].grid(color='black')
    ax[1, 0].set_facecolor('white')

    r_sq = get_r_squared(valid_der_truth, valid_der_pred)
    
    ax[1, 1].scatter(valid_der_truth, valid_der_pred, c = colors[1])
    ax[1, 1].plot(x, y, c = scatter_color, linewidth=linewidth)
    ax[1, 1].set_title("Valid Dataset")
    ax[1, 1].set_xlabel("Truth Derivatives")
    ax[1, 1].set_ylabel("Pred Derivatives")   
    ax[1, 1].annotate(f"$R^2$: {r_sq: .2f}", (-10, 5), fontsize=an_fontsize, bbox= bbox)
    ax[1, 1].grid(color='black')
    ax[1, 1].set_facecolor('white')
    
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

def box_plots(nf_amp_diff, nf_angle_diff, ff_amp_diff, ff_angle_diff, dataset, savefig=False):

    plt.style.use("seaborn-v0_8-poster")
    
    num_models = 2       # (integer) This sets how many boxes next to each other per index
    buffer = 0.5         # (float)   This sets the spacing between boxes for the same index
    num_values =  2      # (integer) This sets the number of indices (9 for phase, 6 for curvature)
    position_offset = 1  # (integer) This sets the offset of the indices - increase if neighboring 'groups' are too close
    fontsize = 14

    pos = np.array(range(num_values))*(num_models+position_offset) 
    fig,ax = plt.subplots(1,1,figsize=(8,5))

    nf_amp_diff = np.mean(nf_amp_diff, axis=(1,2))
    nf_angle_diff = np.mean(nf_angle_diff, axis=(1,2))
    #bp0 = ax.boxplot((nf_amp_diff, nf_angle_diff), sym='', positions = pos - buffer, widths=0.6)
    vp0 = ax.violinplot((nf_amp_diff, nf_angle_diff), positions = pos - buffer, widths=0.6, showmeans=True, showmedians=True, showextrema=True, points=len(nf_amp_diff))

    ff_amp_diff = np.mean(ff_amp_diff, axis=(1,2))
    ff_angle_diff = np.mean(ff_angle_diff, axis=(1,2))
    #bp1 = ax.boxplot((ff_amp_diff, ff_angle_diff), sym='', positions = pos + buffer, widths=0.6)
    vp1 = ax.violinplot((ff_amp_diff, ff_angle_diff), positions = pos + buffer, widths=0.6, showmeans=True, showmedians=True, showextrema=True, points=len(nf_amp_diff))

    # set_box_color(bp0, colors[0], 'solid')
    # set_box_color(bp1, colors[1], 'solid')

    set_violin_color(vp0, colors[0])
    set_violin_color(vp1, colors[1])
    
    ax.tick_params(axis='both', labelsize=fontsize)
    
    ax.set_xticks(range(0, 2*(num_models+position_offset), (num_models+position_offset)))
    ax.set_xticklabels(['Near Field', 'Far Field'])
    ax.set_xlim(-(buffer*2), 2*((num_models)+position_offset)-(buffer*2))
    ax.set_ylim(-0.1,4)
    ax.grid(axis='x', which='both', linewidth=0)  # Set linewidth to 0 to hide the grid lines
    ax.grid(axis='y', which='both', linewidth=1, linestyle='--', color='black', alpha=0.7)  # Set linewidth to 0 to hide the grid lines

    ax.set_ylabel("Average Absolute Difference")
    
    #Gotta do this to get the legend
    #Plots empty lines with the correct color and linestyle to match. We will use these in the legend. 
    ln0, = ax.plot([], c=colors[0], label='Amplitude', linestyle = 'solid')
    ln1, = ax.plot([], c=colors[1], label='Phase', linestyle = 'solid')
    leg = ax.legend(handles = [ln0, ln1] , loc='upper left', frameon = True)

    leg.get_frame().set_edgecolor('gray')

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

    nf_resim = []
    for folder in os.listdir(path_resims):
        if folder_name in folder:
            if target in folder:
                resim = pickle.load(open(os.path.join(path_resims,folder),"rb"))
                nf_resim.append(resim['near_fields']['grating_ey'])
    return nf_resim


def get_ff_resim(nf_resim):
    params = yaml.load(open('../config.yaml', 'r'), Loader=yaml.FullLoader)
    pm = parameter_manager.Parameter_Manager(params = params)
    
    prop = propagator.Propagator(pm.params_propagator)

    nf_resim = np.array(nf_resim)
    ff_resim = prop(torch.tensor(nf_resim))

    return ff_resim

def plot_dft_fields(truth, recon, resim, target, batch=True, idx=None, savefig=False):

    cmap = 'turbo'

    if batch is True:

        titles = ['Truth Intensity', 'Truth Phase', 'Recon Intensity', 'Recon Phase', 'Resim Intensity', 
        'Resim Phase']
        #### index into truth, recon, and resim to get intensity and phase ###
        #### also vmin and vmax ##############################################
        data_map = {
            
            0: (lambda t: t[0]**2, 0, None),
            1: (lambda t: t[1], -torch.pi, torch.pi),
            2: (lambda rec: rec[0]**2, 0, None),
            3: (lambda rec: rec[1], -torch.pi, torch.pi),
            4: (lambda res: np.abs(res)**2, 0, None),
            5: (lambda res: np.angle(res), -torch.pi, torch.pi),
            
        }
        
        fig, ax = plt.subplots(truth.shape[0], len(data_map.keys()), figsize=(10,truth.shape[0]*2))
        fig.suptitle(target)
        
        for i, (t, rec, res) in enumerate(zip(truth, recon, resim)):
    
            for j in range(len(data_map.keys())):
                
                data_func, vmin, vmax = data_map[j]
                data = data_func(t if j < 2 else rec if j < 4 else res)
    
                im = ax[i][j].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    
                divider = make_axes_locatable(ax[i][j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=6)
    
                ax[i][j].set_title(titles[j], fontsize=8)
                ax[i][j].grid(False)
                ax[i][j].axis('off')
                
    else:
        
        fig, ax = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(5,8), gridspec_kw={'width_ratios': [0.03, 1, 1]})
        fig.suptitle(target)
        
        abs_truth = truth[idx][0]
        angle_truth = truth[idx][1]
        abs_recon = recon[idx][0]
        angle_recon = recon[idx][1]
        abs_resim = np.abs(resim[idx])
        angle_resim = np.angle(resim[idx])
    
        ax[0][1].imshow(abs_truth**2, cmap = cmap, vmin = 0)    
        ax[0][2].imshow(angle_truth, cmap = cmap, vmin = -torch.pi, vmax = torch.pi)
        
        ax[1][1].imshow(abs_recon**2, cmap = cmap, vmin = 0)
        ax[1][2].imshow(angle_recon, cmap = cmap, vmin = -torch.pi, vmax = torch.pi)
        
        ax[2][1].imshow(abs_resim**2, cmap = cmap, vmin = 0)
        ax[2][2].imshow(angle_resim, cmap = cmap, vmin = -torch.pi, vmax = torch.pi)
    
        column_titles = ['Intensity', 'Phase']
        row_titles = ['Original/Truth', 'Recon', 'Resim']  # these row titles won't appear with axes off
    
        for j in range(2):
            ax[0][j+1].set_title(column_titles[j], fontsize=10)
        
        for i in range(3):
            #ax[i][0].set_ylabel(row_titles[i], fontsize=10)
            ax[i][0].text(-0.5, 0.5, row_titles[i], transform = ax[i][0].transAxes, rotation=90, ha='center', va='center', fontsize=10)
            ax[i][0].axis("off")
            
        for i in range(3):
            for j in range(1, 3):
                ax[i][j].grid(False)
                ax[i][j].axis("off")
        
    fig.tight_layout()
    
    if savefig == True:
        flag = 'batch' if batch else f'single_sample_idx_{idx}'
        fig.savefig(f'other_plots/{target}_{flag}.pdf')


