import ml_load_script_netcdf as ml_load
import post_processing_figures as post_processing_figures

import numpy as np
import time
import pickle
import os
import math
import random
import sklearn
from dask.diagnostics import ProgressBar
import psutil

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torch import nn, optim
from netCDF4 import Dataset
import netCDF4 as nc
from sklearn.metrics import r2_score
import pdb
import xarray as xr
import math
import zarr
from tqdm import tqdm
from torchview import draw_graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from IPython.display import HTML
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import skimage
import plotly.graph_objects as go

from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib import ticker
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib import transforms
import yaml


def convert_to_tensors(dask_array, desc):
    """Primitive version of the conversion of the data to tensors by batches."""
    # delayed is a form of xarray lazy loading for the data array
    delayed_chunks = dask_array.to_delayed().flatten()
    tensor_list = []
    
    for chunk in tqdm(delayed_chunks, desc=desc, unit="chunk"):
        np_chunk = chunk.compute()
        tensor_chunk = torch.tensor(np_chunk, dtype=torch.float32)
        tensor_list.append(tensor_chunk)
    
    return tensor_list

def get_available_memory():
    """
    Get available memory in bytes using psutil. Helps to determine the chunk size
    """
    return psutil.virtual_memory().available

def calculate_chunk_size(array_shape, dtype_size, chunk_dim='sample', memory_fraction=0.5):
    """
    Calculate the optimal chunk size for a Dask array.
    
    Parameters:
    - array_shape: tuple, shape of the array (vertical_level, sample)
    - dtype_size: int, size of the data type in bytes (e.g., 4 bytes for float32)
    - chunk_dim: str, dimension to primarily chunk ('sample' or 'vertical_level')
    - memory_fraction: float, fraction of available memory to use for chunks
    
    Returns:
    - chunks: tuple, optimal chunk sizes
    """
    available_memory = get_available_memory()  # Get available memory dynamically
    available_memory *= memory_fraction  # Use a fraction of available memory

    total_elements = array_shape[0] * array_shape[1]
    target_chunk_elements = available_memory // dtype_size
    
    if chunk_dim == 'sample':
        chunk_size_samples = target_chunk_elements // array_shape[0]
        chunk_size_samples = max(1, min(array_shape[1], chunk_size_samples))
        return (array_shape[0], chunk_size_samples)
    else:
        chunk_size_levels = target_chunk_elements // array_shape[1]
        chunk_size_levels = max(1, min(array_shape[0], chunk_size_levels))
        return (chunk_size_levels, array_shape[1])


# ---  build random forest or neural net  ---
def train_wrapper(config_file):

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    config_id=config.get('id')
    training_data_path=config.get('training_data_path')
    test_data_path=config.get('test_data_path')
    weights_path=config.get('weights_path')
    save_path=config.get('save_path')
    single_file=config.get('single_file')
    input_vert_vars=config.get('input_vert_vars') 
    output_vert_vars=config.get('output_vert_vars') 
    machine_type=config.get('machine_type')
    poles=config.get('poles')
    training_data_volume=config.get('training_data_volume')
    test_data_volume=config.get('test_data_volume')
    layer_sizes=config.get('layer_sizes')
    nametag=config.get('nametag')
    random_seed=config.get('random_seed')
    z_dim=config.get('z_dim')
    epochs=config.get('epochs')
    lr=config.get('lr')
    dtype_size=config.get('dtype_size')
    mem_frac=config.get('mem_frac')
    batch_size=config.get('batch_size') 
    rewight_outputs=config.get('rewight_outputs')
    train_new_model=config.get('train_new_model')
    plot_analysis=config.get('plot_analysis')
    restrict_land_frac=config.get('restrict_land_frac')

    # generate unique tag to save the experiment
    nametag = "EXPERIMENT_"+str(config_id)+"_"+nametag + "_machine_type_"+machine_type+"_use_poles_"+str(poles)+"_physical_weighting_"+str(rewight_outputs)+"_epochs_"+str(epochs)+"_tr_data_percent_"+str(int(training_data_volume))+"_num_hidden_layers_"+str(len(layer_sizes))
 
    if os.path.isdir(save_path+nametag):
        pass
    else:
        os.mkdir(save_path+nametag)

    if rewight_outputs == True:
        weights = xr.open_dataset(weights_path).norms.values
    else:
        weights=None
    # scale the data, and reformat to (sample, input or output size)
    final_train_inputs, final_test_inputs, final_train_outputs, final_test_outputs, scaled_inputs, scaled_outputs = ml_load.LoadDataStandardScaleData_v4(
                                                        traindata=training_data_path,
                                                       testdata=test_data_path,
                                                       input_vert_vars=input_vert_vars,
                                                       output_vert_vars=output_vert_vars,
                                                        single_file=single_file,
                                                        z_dim=z_dim,
                                                        poles=poles,
                                                        training_data_volume=training_data_volume,
                                                        test_data_volume=test_data_volume,
                                                       weights=weights,
                                                      restrict_land_frac=restrict_land_frac,
                                                       
    )

    # to ensure unique starting point of NN weights
    set_random_seeds(seed=random_seed)

    input_size = final_train_inputs.shape[1]
    output_size = final_train_outputs.shape[1]

    dtype_size = 4  # float32 size in bytes
    mem_frac = 0.1

    # calculate a chunk size based on memory aviaible in selected queue (e.g. RM-512)
    chunk_size_train_inputs = calculate_chunk_size(final_train_inputs.shape, dtype_size, chunk_dim='sample', memory_fraction=mem_frac)
    chunk_size_train_outputs = calculate_chunk_size(final_train_outputs.shape, dtype_size, chunk_dim='sample', memory_fraction=mem_frac)
    chunk_size_test_inputs = calculate_chunk_size(final_test_inputs.shape, dtype_size, chunk_dim='sample', memory_fraction=mem_frac)
    chunk_size_test_outputs = calculate_chunk_size(final_test_outputs.shape, dtype_size, chunk_dim='sample', memory_fraction=mem_frac)

    # rechunk the data to fit wthin memory limits
    final_train_inputs = final_train_inputs.rechunk(chunk_size_train_inputs)
    final_train_outputs = final_train_outputs.rechunk(chunk_size_train_outputs)
    final_test_inputs = final_test_inputs.rechunk(chunk_size_test_inputs)
    final_test_outputs = final_test_outputs.rechunk(chunk_size_test_outputs)

    print("Creating Map Blocks...")
    # prepare data to be in numpy format in delayed loading blocks
    train_inputs_np = final_train_inputs.map_blocks(lambda x: x, dtype=final_train_inputs.dtype)
    train_outputs_np = final_train_outputs.map_blocks(lambda x: x, dtype=final_train_outputs.dtype)
    test_inputs_np = final_test_inputs.map_blocks(lambda x: x, dtype=final_test_inputs.dtype)
    test_outputs_np = final_test_outputs.map_blocks(lambda x: x, dtype=final_test_outputs.dtype)

    # converts the data to tensors by chunks -- time consuming (I think can be ~1 hour)
    # idea - reformat to do this by baqtch size rather than chunk
    pbar = ProgressBar()
    pbar.register()
    temp = convert_to_tensors(train_inputs_np, "Converting train inputs")
    train_inputs_tensors = torch.cat(temp, dim=1)
    pbar.unregister()
    

    pbar = ProgressBar()
    pbar.register()
    temp = convert_to_tensors(train_outputs_np, "Converting train outputs")
    train_outputs_tensors = torch.cat(temp, dim=1)
    pbar.unregister()
    
    pbar = ProgressBar()
    pbar.register()
    temp = convert_to_tensors(test_inputs_np, "Converting test inputs")
    test_inputs_tensors = torch.cat(temp, dim=1)
    pbar.unregister()
    
    pbar = ProgressBar()
    pbar.register()
    temp = convert_to_tensors(test_outputs_np, "Converting test outputs")
    test_outputs_tensors = torch.cat(temp, dim=1)
    pbar.unregister()

    # format the data for a pytorch NN
    train_dataset = torch.utils.data.TensorDataset(train_inputs_tensors, train_outputs_tensors)
    test_dataset = torch.utils.data.TensorDataset(test_inputs_tensors, test_outputs_tensors)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the model, loss function, and optimizer
    model = CustomNN(input_size, layer_sizes, output_size)

    # Use DataParallel for multiple GPUs -- this allows for the option of future paralleization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # https://github.com/mert-kurttutan/torchview
    if os.path.isdir(save_path+nametag+"/Design/"):
        pass
    else:
        os.mkdir(save_path+nametag+"/Design/")
    path_for_design = save_path + "/"+nametag+"/Design/"
    model_graph = draw_graph(model, 
                                input_size=(batch_size, input_size),
                                graph_name=nametag,
                                save_graph=True,
                                directory=path_for_design,
                                    )


    # in my opinion, logical stardard loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model and get the losses
    name = "epochs_"+str(epochs)+"_lr_"+str(lr)+"_inputs_"+str(input_size)+"_outputs_"+str(output_size)+"_"
    savename = save_path+nametag+"/Saved_Models/"
    
    if train_new_model is True:
        train_losses, test_losses, train_accuracies, test_accuracies, avg_epoch_time = new_train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=epochs, device=device)

    # Plot the losses and accuracies
        plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, avg_epoch_time, name, save_path, nametag)

        # save the weights of your trained model
        if os.path.isdir(savename):
            pass
        else:
            os.mkdir(savename)
        
        torch.save(model.state_dict(), savename+"weights.pkl")
        save_model_to_netcdf(model, savename+"weights.nc")

    else:
        model.load_state_dict(torch.load(savename+"weights.pkl"))

        # Set the model to evaluation mode
        model.eval()


    if plot_analysis is True:
        
        predictions = get_model_predictions(model, test_loader, device=device)

        if isinstance(z_dim, tuple):
            # Do something if it's a tuple
            my_z = z_dim[0]
        else:
            # Do something else if it's not a tuple
            my_z = z_dim
        
        truths, preds = ml_load.unscale_all(truth=final_test_outputs, 
                                            predictions=predictions, 
                                            z_dim=my_z, 
                                            means=scaled_outputs['mean'], 
                                            stds=scaled_outputs['std'], 
                                            weights=weights,
                                           )
        
        post_processing_figures.main_plotting(truth=truths,
                                              pred=preds, 
                                              raw_data=single_file,
                                              z_dim=my_z,
                                             var_names=output_vert_vars,
                                              save_path=save_path,
                                              nametag=nametag,
                                             )

    return "Training Complete"

# Function to set random seeds
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the customizable neural network architecture
class CustomNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(CustomNN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Function to train the model
def new_train_model(model, train_loader, test_loader, criterion, optimizer, device="cuda", num_epochs=10):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time() 
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, targets_max = torch.max(targets.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets_max).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(100 * correct_train / total_train)

        model.eval()
        total_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, targets_max = torch.max(targets.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets_max).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracies.append(100 * correct_test / total_test)

        epoch_end_time = time.time()  # End time
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Test Accuracy: {test_accuracies[-1]:.2f}%")

    return train_losses, test_losses, train_accuracies, test_accuracies, np.mean(epoch_times)


def save_model_to_netcdf(model, filename):
    # Extract state_dict
    state_dict = model.state_dict()
    
    # Create a new NetCDF file
    ds = nc.Dataset(filename, 'w', format='NETCDF4')
    
    # Create dimensions
    ds.createDimension('layers', len(state_dict))
    
    # Save each parameter
    for i, (key, value) in enumerate(state_dict.items()):
        # Convert to numpy array
        np_value = value.numpy()
        
        # Determine shape and create variable
        dim_name = f'dim_{i}'
        ds.createDimension(dim_name, np_value.size)
        var = ds.createVariable(key, np_value.dtype, (dim_name,))
        
        # Write data
        var[:] = np_value.flatten()
    
    # Close the NetCDF file
    ds.close()



# Function to plot the losses and accuracies
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epoch_time, name, save_path, nametag):
    if os.path.isdir(save_path+nametag+'/Loss_Curves'):
        pass
    else:
        os.mkdir(save_path+nametag+'/Loss_Curves')
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))

    # Subplot 1: Training and Test Loss
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss vs. Epoch')
    # Add textbox
    textstr1 = "avg. time per epoch: "+str(int(epoch_time/60))+' minutes'
    bbox_props1 = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5)
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=bbox_props1)

    # Subplot 2: Training and Test Accuracy
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, train_accuracies, label='Train Accuracy')
    ax2.plot(epochs, test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy vs. Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path+nametag+"/Loss_Curves/Model_Losses.pdf")


def get_model_predictions(model, test_loader, device="cuda"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return np.vstack(predictions)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ml_train_nn_zarr.py <config_file.yaml>")
    else:
        train_wrapper(sys.argv[1])


