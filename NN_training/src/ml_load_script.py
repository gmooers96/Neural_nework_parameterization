import numpy as np
import math
import zarr
import dask.array as da


def train_percentile_calc(percent, ds):
    sample = len(ds.sample)
    lon = len(ds.lon)
    lat = len(ds.lat)
    times = int(sample / (lon*lat))
    proportion = math.ceil(times*percent/100.)
    splicer = int(proportion*lon*lat)
    return splicer

def standardize(arr, mean=None, std=None):
    if mean is None:
        mean = arr.mean()
    if std is None:
        std = arr.std()
    return (arr - mean) / std, mean, std

def unstandardize(array, mean, std, weight=None):
    if weight is None:
        return (array * std) + mean
    else:
        return ((array * std) + mean) / weight

def slice_by_lat_index(lats, n_files, n_x):
    absolute_differences = np.abs(lats - 70.)
    nh_closest_index = np.argmin(absolute_differences)
    absolute_differences = np.abs(lats + 70.0)
    sh_closest_index = np.argmin(absolute_differences)

    slice_size = n_files * n_x
    start_idx = sh_closest_index * slice_size
    end_idx = nh_closest_index * slice_size

    return int(start_idx), int(end_idx)

def convert_dict_to_array(my_dict):
    dask_arrays = []
    for key, array in my_dict.items():
        if array.ndim == 1:
            array = array[np.newaxis, :]
        dask_arrays.append(array)

    # Concatenate along axis 0
    combined_array = da.concatenate(dask_arrays, axis=0)
    return combined_array


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
                                 memory_limit_gb=15):
    """
    Load and standardize data from Zarr files efficiently using calculated chunk size.
    :param traindata: Path to the training data Zarr file.
    :param testdata: Path to the testing data Zarr file.
    :param input_vert_vars: List of input vertical variables.
    :param output_vert_vars: List of output vertical variables.
    :param single_file: Path to a file containing latitude data for slicing.
    :param z_dim: Vertical dimension size.
    :param poles: Boolean flag for handling polar data.
    :param training_data_volume: Percentage of training data to use.
    :param test_data_volume: Percentage of testing data to use.
    :param weights: Optional weights for scaling.
    :param memory_limit_gb: Memory limit in gigabytes for chunk size calculation.
    :return: Scaled input and output dictionaries.
    """

    # Load Zarr groups (assuming hierarchical structure)
    train_store = zarr.open_group(traindata, mode='r')
    test_store = zarr.open_group(testdata, mode='r')
    
    training_data_percentage = train_percentile_calc(training_data_volume, train_store)
    test_data_percentage = train_percentile_calc(test_data_volume, test_store)
    
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

        train_dask_array = da.from_zarr(train_store[inputs])
        test_dask_array = da.from_zarr(test_store[inputs])
        
        mean = train_dask_array.mean()
        std = train_dask_array.std()

        # Apply scaling to the test array
        scaled_train_dask_array = (train_dask_array - mean) / std
        scaled_test_dask_array = (test_dask_array - mean) / std

        if poles is False:

            lats = train_store['lat']
            lons = train_store['lon']
            num_lats = len(lats)
            num_lons = len(lons)
            num_files_train = len(train_store['sample'])
            num_files_test = len(test_store['sample'])
            num_train_hours = int(num_files_train / (num_lats*num_lons))
            num_test_hours = int(num_files_test / (num_lats*num_lons))
            
            start_idx_train, end_idx_train = slice_by_lat_index(lats[:], num_train_hours, num_lons)
            start_idx_test, end_idx_test = slice_by_lat_index(lats[:], num_test_hours, num_lons)
            scaled_train_dask_array[:start_idx_train] = 0
            scaled_train_dask_array[end_idx_train:] = 0
            scaled_test_dask_array[:start_idx_test] = 0
            scaled_test_dask_array[end_idx_test:] = 0
            
        if len(scaled_train_dask_array.shape) == 2:
            scaled_train_dask_array = scaled_train_dask_array[:, :training_data_percentage]
            scaled_test_dask_array = scaled_test_dask_array[:, :test_data_percentage]
        if len(scaled_train_dask_array.shape) == 1:
            scaled_train_dask_array = scaled_train_dask_array[:training_data_percentage]
            scaled_test_dask_array = scaled_test_dask_array[:test_data_percentage]

        scaled_inputs['train'][inputs] = scaled_train_dask_array
        scaled_inputs['mean'][inputs] = mean
        scaled_inputs['std'][inputs] = std
        scaled_inputs['test'][inputs] = scaled_test_dask_array

    for outputs in output_vert_vars:

        train_dask_array = da.from_zarr(train_store[outputs])
        test_dask_array = da.from_zarr(test_store[outputs])
        
        mean = train_dask_array.mean()
        std = train_dask_array.std()

        # Apply scaling to the test array
        scaled_train_dask_array = (train_dask_array - mean) / std
        scaled_test_dask_array = (test_dask_array - mean) / std
            
        if len(scaled_train_dask_array.shape) == 2:
            scaled_train_dask_array = scaled_train_dask_array[:, :training_data_percentage]
            scaled_test_dask_array = scaled_test_dask_array[:, :test_data_percentage]
        if len(scaled_train_dask_array.shape) == 1:
            scaled_train_dask_array = scaled_train_dask_array[:training_data_percentage]
            scaled_test_dask_array = scaled_test_dask_array[:test_data_percentage]

        scaled_outputs['train'][outputs] = scaled_train_dask_array
        scaled_outputs['mean'][outputs] = mean
        scaled_outputs['std'][outputs] = std
        scaled_outputs['test'][outputs] = scaled_test_dask_array

    final_train_inputs = convert_dict_to_array(scaled_inputs['train'])
    final_test_inputs = convert_dict_to_array(scaled_inputs['test'])
    final_train_outputs = convert_dict_to_array(scaled_outputs['train'])
    final_test_outputs = convert_dict_to_array(scaled_outputs['test'])
    return final_train_inputs, final_test_inputs, final_train_outputs, final_test_outputs, scaled_inputs, scaled_outputs
    
