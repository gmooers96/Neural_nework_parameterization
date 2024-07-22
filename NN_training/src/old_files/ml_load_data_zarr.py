import numpy as np
from sklearn import preprocessing, metrics
import sklearn
import scipy.stats
import pickle
import warnings
import pandas as pd
from netCDF4 import Dataset
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torch import nn, optim
import xarray as xr
import pdb
import math
import zarr
import gc

def train_percentile_calc(percent, ds):
    sample = len(ds.sample)
    lon = len(ds.lon)
    lat = len(ds.lat)
    times = int(sample / (lon*lat))
    proportion = math.ceil(times*percent/100.)
    splicer = int(proportion*lon*lat)
    return splicer


# Define a custom scaling function
def standardize(arr, mean=None, std=None):
    if mean is None:
        mean = arr.mean()
    if std is None:
        std = arr.std()
    return (arr - mean) / std, mean, std


def unstandardize(array, mean, std, weight=None):
    if weight is None:
        return (array*std)+mean
    else:
        return ((array*std)+mean) / weight

def unscale_all(truth_dict, pred_array, output_vars, z_dim=49, weights=None):
    pred_array = np.swapaxes(pred_array, 0 , 1)
    unscaled_predictions_array = np.empty(pred_array.shape)
    unscaled_truths = {}
    for i in range(len(output_vars)):
        mean = truth_dict['mean'][output_vars[i]]
        std = truth_dict['std'][output_vars[i]]
        if weights is None:
            unscaled_predictions_array[z_dim*i:z_dim*(i+1):,:] = unstandardize(pred_array[z_dim*i:z_dim*(i+1):, :], mean, std)
            unscaled_truths[output_vars[i]] = unstandardize(truth_dict['test'][output_vars[i]], mean, std)
        else:
            unscaled_predictions_array[z_dim*i:z_dim*(i+1):,:] = unstandardize(pred_array[z_dim*i:z_dim*(i+1):, :], mean, std, weight=weights[i])
            unscaled_truths[output_vars[i]] = unstandardize(truth_dict['test'][output_vars[i]], mean, std, weight=weights[i])

    unscaled_truth_array = np.vstack([v for k, v in unscaled_truths.items() if isinstance(v, np.ndarray)])

    return unscaled_truth_array, unscaled_predictions_array


def slice_by_lat_index(lats, n_files, n_x):

    absolute_differences = np.abs(lats - 70.)
    nh_closest_index = np.argmin(absolute_differences)
    absolute_differences = np.abs(lats - 70*-1.0)
    sh_closest_index = np.argmin(absolute_differences)
    
    slice_size = n_files * n_x
    start_idx = sh_closest_index * slice_size
    end_idx = nh_closest_index + slice_size

    return int(start_idx), int(end_idx)
        

def LoadDataStandardScaleData_v4(traindata,
                                 testdata,
                                 input_vert_vars,
                                 output_vert_vars,
                                 single_file,
                                 z_dim,
                                 poles=True,
                                 training_data_volume=10.0,
                                 test_data_volume=50.0,
                                 weights=None,
                                ):
    """

    TODO: gmooers
    """
    #load in the training data
    train_variables = zarr.open(traindata, mode='r')
    test_variables = zarr.open(testdata, mode='r')
    training_data_percentage = train_percentile_calc(training_data_volume, train_variables)
    test_data_percentage = train_percentile_calc(test_data_volume, test_variables)

    scaled_inputs = {
        'train': {},
        'mean': {},
        'std': {},
        'test': {}
    }

    scaled_outputs = {
        'train': {},
        'mean': {},
        'std': {},
        'test': {}
    }

    for inputs in input_vert_vars:
        #print("input:", inputs)
        train_slice = train_variables[inputs][...,:training_data_percentage]
        #print("train spliced", inputs)
        test_slice = test_variables[inputs][...,:test_data_percentage]
        if poles is False:
            single_data = xr.open_dataset(single_file)
            lats = single_data.lat.values
            n_x = len(single_data.lon.values)
            if len(train_slice.shape) == 2:
                axis = 1
            else:
                axis=0
            n_files = train_slice.shape[axis]/(n_x*len(lats))
            start_ind_train, end_ind_train = slice_by_lat_index(lats, n_files, n_x)
            n_files = test_slice.shape[axis]/(n_x*len(lats))
            start_ind_test, end_ind_test = slice_by_lat_index(lats, n_files, n_x)
            train_slice[...,:start_ind_train] = 0
            train_slice[...,end_ind_train:] = 0
            test_slice[...,:start_ind_train] = 0
            test_slice[...,end_ind_train:] = 0
            
        #print("test loaded", inputs)
        scaled_array_train, mean, std = standardize(train_slice)
        #print("scaled train")
        scaled_array_test, mean, std = standardize(test_slice, mean=mean, std=std)
        #print("scaled test")
        if len(scaled_array_train.shape) == 1:
            scaled_array_train = np.expand_dims(scaled_array_train, axis=0)
            scaled_array_test = np.expand_dims(scaled_array_test, axis=0)
        if scaled_array_train.shape[0] > 10:
            scaled_array_train = scaled_array_train[:z_dim]
            scaled_array_test = scaled_array_test[:z_dim]
        scaled_inputs['train'][inputs] = scaled_array_train
        scaled_inputs['mean'][inputs] = mean
        scaled_inputs['std'][inputs] = std
        #print("scaled train in dict")
        scaled_inputs['test'][inputs] = scaled_array_test
        #print("scaled test in dict")
        del scaled_array_train, mean, std, scaled_array_test, train_slice, test_slice
        gc.collect()

    count = 0
    for outputs in output_vert_vars:
        #print("output:", outputs)
        train_slice = train_variables[outputs][...,:training_data_percentage]
        test_slice = test_variables[outputs][...,:test_data_percentage]
        scaled_array_train, mean, std = standardize(train_slice)
        scaled_array_test, mean, std = standardize(test_slice, mean=mean, std=std)
        if len(scaled_array_train.shape) == 1:
            scaled_array_train = np.expand_dims(scaled_array_train, axis=0)
            scaled_array_test = np.expand_dims(scaled_array_test, axis=0)
        if weights is not None:
            scaled_array_train = scaled_array_train*weights[count]
            scaled_array_test = scaled_array_test*weights[count]
            count = count+1
        if scaled_array_train.shape[0] > 10:
            scaled_array_train = scaled_array_train[:z_dim]
            scaled_array_test = scaled_array_test[:z_dim]
        scaled_outputs['mean'][outputs] = mean
        scaled_outputs['std'][outputs] = std
        scaled_outputs['train'][outputs] = scaled_array_train
        scaled_outputs['test'][outputs] = scaled_array_test
        del scaled_array_train, mean, std, scaled_array_test, train_slice, test_slice
        gc.collect()

    return scaled_inputs, scaled_outputs



def LoadDataStandardScaleData_v5(traindata,
                                 testdata,
                                 input_vert_vars,
                                 output_vert_vars,
                                 single_file,
                                 z_dim,
                                 poles=True,
                                 training_data_volume=10.0,
                                 test_data_volume=50.0,
                                 weights=None,
                                ):
    """

    TODO: gmooers
    """
    #load in the training data
    train_variables = zarr.open(traindata, mode='r')
    test_variables = zarr.open(testdata, mode='r')
    training_data_percentage = train_percentile_calc(training_data_volume, train_variables)
    test_data_percentage = train_percentile_calc(test_data_volume, test_variables)

    scaled_inputs = {
        'train': {},
        'mean': {},
        'std': {},
        'test': {}
    }

    scaled_outputs = {
        'train': {},
        'mean': {},
        'std': {},
        'test': {}
    }

    for inputs in input_vert_vars:
        #print("input:", inputs)
        train_slice = train_variables[inputs][...,:training_data_percentage]
        #print("train spliced", inputs)
        test_slice = test_variables[inputs][...,:test_data_percentage]
        if poles is False:
            single_data = xr.open_dataset(single_file)
            lats = single_data.lat.values
            n_x = len(single_data.lon.values)
            if len(train_slice.shape) == 2:
                axis = 1
            else:
                axis=0
            n_files = train_slice.shape[axis]/(n_x*len(lats))
            start_ind_train, end_ind_train = slice_by_lat_index(lats, n_files, n_x)
            n_files = test_slice.shape[axis]/(n_x*len(lats))
            start_ind_test, end_ind_test = slice_by_lat_index(lats, n_files, n_x)
            train_slice[...,:start_ind_train] = 0
            train_slice[...,end_ind_train:] = 0
            test_slice[...,:start_ind_train] = 0
            test_slice[...,end_ind_train:] = 0
            
        #print("test loaded", inputs)
        scaled_array_train, mean, std = standardize(train_slice)
        #print("scaled train")
        scaled_array_test, mean, std = standardize(test_slice, mean=mean, std=std)
        #print("scaled test")
        if len(scaled_array_train.shape) == 1:
            scaled_array_train = np.expand_dims(scaled_array_train, axis=0)
            scaled_array_test = np.expand_dims(scaled_array_test, axis=0)
        if scaled_array_train.shape[0] > 10:
            scaled_array_train = scaled_array_train[:z_dim]
            scaled_array_test = scaled_array_test[:z_dim]
        scaled_inputs['train'][inputs] = scaled_array_train
        scaled_inputs['mean'][inputs] = mean
        scaled_inputs['std'][inputs] = std
        #print("scaled train in dict")
        scaled_inputs['test'][inputs] = scaled_array_test
        #print("scaled test in dict")
        del scaled_array_train, mean, std, scaled_array_test, train_slice, test_slice
        gc.collect()

    count = 0
    for outputs in output_vert_vars:
        #print("output:", outputs)
        train_slice = train_variables[outputs][...,:training_data_percentage]
        test_slice = test_variables[outputs][...,:test_data_percentage]
        scaled_array_train, mean, std = standardize(train_slice)
        scaled_array_test, mean, std = standardize(test_slice, mean=mean, std=std)
        if len(scaled_array_train.shape) == 1:
            scaled_array_train = np.expand_dims(scaled_array_train, axis=0)
            scaled_array_test = np.expand_dims(scaled_array_test, axis=0)
        if weights is not None:
            scaled_array_train = scaled_array_train*weights[count]
            scaled_array_test = scaled_array_test*weights[count]
            count = count+1
        if scaled_array_train.shape[0] > 10:
            scaled_array_train = scaled_array_train[:z_dim]
            scaled_array_test = scaled_array_test[:z_dim]
        scaled_outputs['mean'][outputs] = mean
        scaled_outputs['std'][outputs] = std
        scaled_outputs['train'][outputs] = scaled_array_train
        scaled_outputs['test'][outputs] = scaled_array_test
        del scaled_array_train, mean, std, scaled_array_test, train_slice, test_slice
        gc.collect()

    return scaled_inputs, scaled_outputs



















