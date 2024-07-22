import xarray as xr
import dask
import numpy as np

ds = xr.open_mfdataset("/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/P**[07].nc",
                      concat_dim="sample")

ds_weight = xr.open_mfdataset("/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/P**w8s.nc",
                      concat_dim="sample")

ds_weight.to_netcdf("/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Numpy_Full_Training_dataset_weight.nc")

norms = ds_weight.norms.values

my_norm = np.mean(norms, axis=0)

my_weight_dict = {}
my_weight_dict["norms"] = (("norm"), my_norm)
ds_weight_new = xr.Dataset(
        my_weight_dict,
            coords={
                "norm":np.arange(1,6,1),
            },
        )

ds_weight_new.to_netcdf('/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/All_Weights.nc')
ds.to_netcdf('/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/Numpy_Full_Training_dataset.nc')