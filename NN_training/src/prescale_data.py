import xarray as xr

# Open the Zarr dataset with xarray and Dask
training_data_path: "/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/New_Full_Training_dataset.zarr"
test_data_path: '/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/New_Full_Test_dataset.zarr'


ds_train = xr.open_zarr('/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/New_Full_Training_dataset.zarr')
ds_est = xr.open_zarr('path/to/your/zarr')

# Function to scale data
def scale(da):
    mean = da.mean()
    std = da.std()
    return (da - mean) / std

# Apply scaling to each variable in the dataset
scaled_ds = ds.map(scale)

# Write the scaled dataset to a new Zarr store
scaled_ds.to_zarr('path/to/scaled/zarr', mode='w', consolidated=True)