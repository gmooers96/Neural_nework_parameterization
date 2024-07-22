import src.ml_load as ml_load
from netCDF4 import Dataset
import netCDF4
import numpy as np
import pickle
import glob
import src.atmos_physics as atmos_physics
import numpy.matlib
import sys
import random 
import pdb
import math
import xarray as xr

from src.train_test_generator_helper_functions import  create_specific_data_string_desc,  calculate_renormalization_factors, get_train_test_split


def build_training_dataset(filepath, 
                           savepath,
                           my_label,
                           train_size=0.9, 
                           n_z_input=74,
                           flag_dict=dict(), 
                           rewight_outputs = False,
                          ):
    """Builds training and testing datasets
    """

    variables = xr.open_mfdataset(filepath)
    
    x = variables.lon  # m
    y = variables.lat  # m
    z = variables.z  # m
    p = variables.p  # hPa
    rho = variables.rho  # kg/m^3
    n_x = x.size
    n_y = y.size
    n_z = z.size
    n_files = len(variables.time)
    
    if flag_dict['land_frac']:
        terra = variables.TERRA[:,:n_z_input]
        
    if flag_dict['sfc_pres']:
        SFC_PRES = variables.SFC_REFERENCE_P
        
    if flag_dict['skt']:
        SKT = variables.SKT
    
    if flag_dict['cos_lat']:
        cos_lat = np.zeros((n_files, n_y, n_x))
        cos_lat[:, :, :] = xr.ufuncs.cos(xr.ufuncs.radians(y.values[None, :, None]))
    
    if flag_dict['sin_lon']:
        sin_lon = np.zeros((n_files, n_y, n_x))
        sin_lon[:, :, :] = xr.ufuncs.sin(xr.ufuncs.radians(x.values[None, None, :]))
    
    adz = xr.zeros_like(z[:n_z_input]) 
    dz = 0.5*(z[0]+z[1]) 
    adz[0] = 1.

    for k in range(1,n_z_input-1): # range doesn't include stopping number
        adz[k] = 0.5*(z[k+1]-z[k-1])/dz

    adz[n_z_input-1] = (z[n_z_input-1]-z[n_z_input-2])/dz
    rho_dz = adz*dz*rho
    
    Tin = variables.TABS_SIGMA[:,:n_z_input] #originally just called tabs
    Qrad = variables.QRAD_SIGMA[:,:n_z_input] / 86400
    
    if flag_dict['qin_feature']:
        qt = (variables.QV_SIGMA[:,:n_z_input] + variables.QC_SIGMA[:,:n_z_input] + variables.QI_SIGMA[:,:n_z_input]) / 1000.0 
    
    qp = variables.QP_SIGMA[:,:n_z_input] / 1000.0
    q_auto_out = -1.0*variables.QP_MICRO_SIGMA[:,:n_z_input] / 1000.0
    qpflux_z_coarse = variables.RHOQPW_SIGMA[:,:n_z_input] / 1000.0
    T_adv_out = variables.T_FLUX_Z_OUT_SUBGRID_SIGMA[:,:n_z_input]     #originally tflux_z
    q_adv_out = variables.Q_FLUX_Z_OUT_SUBGRID_SIGMA[:,:n_z_input] / 1000.0 #originally qtflux_z
    qpflux_z = variables.QP_FLUX_Z_OUT_SUBGRID_SIGMA[:,:n_z_input] / 1000.0 
    w = variables.W[:,:n_z_input]  # m/s
    precip = variables.PREC_SIGMA[:,:n_z_input]  # precipitation flux kg/m^2/s
    cloud_qt_flux = variables.SED_SIGMA[:,:n_z_input] / 1000.0
    cloud_lat_heat_flux = variables.LSED_SIGMA[:,:n_z_input] 
    qpflux_diff_coarse_z = variables.RHOQPS_SIGMA[:,:n_z_input] / 1000.0  # SGS qp flux kg/m^2/s Note that I need this variable
    
    a_pr = 1.0 / (atmos_physics.tprmax - atmos_physics.tprmin)
    omp = np.maximum(0.0, np.minimum(1.0, (Tin - atmos_physics.tprmin) * a_pr))
    fac = (atmos_physics.L + atmos_physics.Lf * (1.0 - omp)) / atmos_physics.cp
    
    q_sed_fluxc_out = ((atmos_physics.L + atmos_physics.Lf) * cloud_qt_flux + cloud_lat_heat_flux) / atmos_physics.Lf
    q_sed_fluxi_out = - (atmos_physics.L * cloud_qt_flux + cloud_lat_heat_flux) / atmos_physics.Lf
    q_sed_flux_tot  = cloud_qt_flux
    
    dfac_dz = np.zeros((n_files, n_z_input, n_y, n_x))
    for k in range(n_z_input - 1):
        kb = max(0, k - 1)
        dfac_dz[:, k, :, :] = (fac[:, k + 1, :, :] - fac[:, k, :, :]) / rho_dz[k, :] * rho[:, k]
        
    Tout = dfac_dz * (qpflux_z_coarse + qpflux_diff_coarse_z - precip) / rho
    
    split_index = get_train_test_split(train_size, n_x, n_files)
    
    data_specific_description = create_specific_data_string_desc(flag_dict)
    
    my_dict_train = {}
    my_dict_test = {}
    
    if flag_dict['Tin_feature']:
        Tin = Tin.transpose("z","lat","time","lon").values
        Tin = np.reshape(Tin, (n_z_input, n_y, n_files*n_x))
        my_dict_train["Tin"] = (("z","lat","sample"), Tin[...,:split_index])
        my_dict_test["Tin"] = (("z","lat","sample"), Tin[...,split_index:])
    
    if flag_dict['qin_feature']:
        qin = qt.transpose("z","lat","time","lon").values
        qin = np.reshape(qin, (n_z_input, n_y, n_files*n_x))
        my_dict_train["qin"] = (("z","lat","sample"), qin[...,:split_index])
        my_dict_test["qin"] = (("z","lat","sample"), qin[...,split_index:])
    
    if flag_dict['predict_tendencies']:
        Tout = Tout.transpose("z","lat","time","lon").values
        Tout = np.reshape(Tout, (n_z_input, n_y, n_files*n_x))
        my_dict_train["Tout"] = (("z","lat","sample"), Tout[...,:split_index])
        my_dict_test["Tout"] = (("z","lat","sample"), Tout[...,split_index:])

        T_adv_out = T_adv_out.transpose("z","lat","time","lon").values
        T_adv_out = np.reshape(T_adv_out, (n_z_input, n_y, n_files*n_x))
        my_dict_train["T_adv_out"] = (("z","lat","sample"), T_adv_out[...,:split_index])
        my_dict_test["T_adv_out"] = (("z","lat","sample"), T_adv_out[...,split_index:])
        
        q_adv_out = q_adv_out.transpose("z","lat","time","lon").values
        q_adv_out = np.reshape(q_adv_out, (n_z_input, n_y, n_files*n_x))
        my_dict_train["q_adv_out"] = (("z","lat","sample"), q_adv_out[...,:split_index])
        my_dict_test["q_adv_out"] = (("z","lat","sample"), q_adv_out[...,split_index:])
        
        q_auto_out = q_auto_out.transpose("z","lat","time","lon").values
        q_auto_out = np.reshape(q_auto_out, (n_z_input, n_y, n_files*n_x))
        my_dict_train["q_auto_out"] = (("z","lat","sample"), q_auto_out[...,:split_index])
        my_dict_test["q_auto_out"] = (("z","lat","sample"), q_auto_out[...,split_index:])

        q_sed_flux_tot = q_sed_flux_tot.transpose("z","lat","time","lon").values
        q_sed_flux_tot = np.reshape(q_sed_flux_tot, (n_z_input, n_y, n_files*n_x))
        my_dict_train["q_sed_flux_tot"] = (("z","lat","sample"), q_sed_flux_tot[...,:split_index])
        my_dict_test["q_sed_flux_tot"] = (("z","lat","sample"), q_sed_flux_tot[...,split_index:])
    
    if flag_dict['land_frac']:
        terra = terra.transpose("z","lat","time","lon").values
        terra = np.reshape(terra, (n_z_input, n_y, n_files*n_x))
        my_dict_train["terra"] = (("z","lat","sample"), terra[...,:split_index])
        my_dict_test["terra"] = (("z","lat","sample"), terra[...,split_index:])
    
    if flag_dict['sfc_pres']:
        sfc_pres = SFC_PRES.transpose("lat","time","lon").values
        sfc_pres = np.reshape(sfc_pres, (n_y, n_files*n_x))
        my_dict_train["sfc_pres"] = (("lat","sample"), sfc_pres[...,:split_index])
        my_dict_test["sfc_pres"] = (("lat","sample"), sfc_pres[...,split_index:])
    
    if flag_dict['skt']:
        skt = SKT.transpose("lat","time","lon").values
        skt = np.reshape(skt, (n_y, n_files*n_x))
        my_dict_train["skt"] = (("lat","sample"), skt[...,:split_index])
        my_dict_test["skt"] = (("lat","sample"), skt[...,split_index:])
    
    if flag_dict['cos_lat']:
        cos_lat = np.expand_dims(cos_lat, axis=0)
        cos_lat = np.moveaxis(cos_lat, 2, 3)
        cos_lat = np.reshape(cos_lat, (1, n_y, -1)).squeeze()
        my_dict_train["cos_lat"] = (("lat","sample"), cos_lat[...,:split_index])
        my_dict_test["cos_lat"] = (("lat","sample"), cos_lat[...,split_index:])

    if flag_dict['sin_lon']:
        sin_lon = np.expand_dims(sin_lon, axis=0)
        sin_lon = np.moveaxis(sin_lon, 2, 3)
        sin_lon = np.reshape(sin_lon, (1, n_y, -1)).squeeze()
        my_dict_train["sin_lon"] = (("lat","sample"), sin_lon[...,:split_index])
        my_dict_test["sin_lon"] = (("lat","sample"), sin_lon[...,split_index:])
    
    
    if rewight_outputs == True:
        my_weight_dict = {}
        norm_list = calculate_renormalization_factors(Tout[...,:split_index],
                                                       T_adv_out[...,:split_index],
                                                       q_adv_out[...,:split_index],
                                                       q_auto_out[...,:split_index],
                                                       q_sed_flux_tot[...,split_index:],
                                                       rho_dz[:,0].values,
                                                      ) 
        my_weight_dict["norms"] = (("norm"), norm_list)
        ds_weight = xr.Dataset(
        my_weight_dict,
            coords={
                "norm":np.arange(1,6,1),
            },
        )
        ds_weight.to_netcdf(savepath + my_label + data_specific_description + "_weight.nc")

    
    
    ds_train = xr.Dataset(
    my_dict_train,
    coords={
        "z": z[:n_z_input].values,
        "lat": y.values,
        "lon": x.values,
        "z_profile": z.values,
        "rho": rho[0,:].values,
        "p": p[0,:].values,
        "sample": np.arange(0,n_files*len(x.values), 1)[:split_index],
    },)
    
    ds_test = xr.Dataset(
    my_dict_test,
    coords={
        "z": z[:n_z_input].values,
        "lat": y.values,
        "lon": x.values,
        "z_profile": z.values,
        "rho": rho[0,:].values,
        "p": p[0,:].values,
        "sample": np.arange(0,n_files*len(x.values), 1)[split_index:],
    },)
    
    ds_train.to_netcdf(savepath + my_label + data_specific_description + "_train.nc")
    ds_test.to_netcdf(savepath + my_label + data_specific_description +"_test.nc")
    
    