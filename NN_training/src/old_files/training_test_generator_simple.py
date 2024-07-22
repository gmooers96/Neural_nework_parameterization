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

from src.train_test_generator_helper_functions import write_netcdf_nn, create_z_grad_plus_surf_var, create_difference_from_surface, vertical_smooth, create_specific_data_string_desc, print_simulation_decription, calculate_renormalization_factors, calculate_diffusion_renormalization_factors, calculate_renormalization_diff_tend, calculate_renormalization_diff_tend_separate_flux, calculate_renormalization_factors_all_diff


def build_training_dataset(expt, start_time, end_time, interval, n_x_samp=5, train_size=0.9, do_shuffle=True,
                           flag_dict=dict(), is_bridges=True,
                           dx=12000 * 16,
                           dy=12000 * 16, rewight_outputs = False):
    """Builds training and testing datasets
    Args:
     expt (str): Experiment name
     interval (int): Number of timesteps between when each file is saved
     start_time (int): First timestep
     end_time (int): Last timestep
     n_x_samp (int): Number of random samples at each y and time step
     flag_dict (dict): including the specific configuration that we want to calculate the outputs for
    """

    if is_bridges == True:  # On aimsir/esker
        base_dir2 = '/ocean/projects/ees220005p/gmooers/'
        base_dir = '/ocean/projects/ees220005p/gmooers/GM_Data/'  
        
    output_dir = base_dir + '/training_data/'

    np.random.seed(123)
    
    filename_wildcard = base_dir + 'DYAMOND2_coars_9216x4608x74_10s_4608_' + str(flag_dict['starting_phase']) + '_' + str(flag_dict['file_zero_time']) + '.atm.3D_resolved.nc4'

    filename = glob.glob(filename_wildcard)
    f = Dataset(filename[0], mode='r') 
    x = f.variables['lon'][:]  # m
    y = f.variables['lat'][:]  # m
    print("y", y.shape)
    print("x", x.shape)
    z = f.variables['z'][:]  # m
    p = f.variables['p'][:]  # hPa
    rho = f.variables['rho'][:]  # kg/m^3
    
    #Extra variables for input vector
    terra = f.variables['TERRA'][:] 
    SFC_PRES = f.variables['SFC_REFERENCE_P'][:]
    SKT = f.variables['SKT'][:]
    n_x = len(x)
    n_y = len(y)
    n_z = len(z)
    n_z_input = flag_dict['input_upper_lev']
    f.close()

    # Initialize
    file_times = np.arange(start_time, end_time + interval, interval)
    n_files = np.size(file_times)

    Tin = np.zeros((n_z_input, n_y, n_x_samp, n_files))
    qin = np.zeros((n_z_input, n_y, n_x_samp, n_files))
    Tout = np.zeros((n_z, n_y, n_x_samp, n_files))
    qout = np.zeros((n_z, n_y, n_x_samp, n_files))

    if flag_dict['cos_lat']:
        cos_lat = np.zeros((n_y, n_x_samp, n_files))
    if flag_dict['sin_lon']:
        sin_lon = np.zeros((n_y, n_x_samp, n_files))
    if flag_dict['sfc_pres']:
        sfc_pres = np.zeros((n_y, n_x_samp, n_files)) 
    if flag_dict['skt']:
        skt = np.zeros((n_y, n_x_samp, n_files))
    if flag_dict['land_frac']:
        land_frac = np.zeros((n_z_input, n_y, n_x_samp, n_files))

    T_adv_out = np.zeros((n_z, n_y, n_x_samp, n_files))
    q_adv_out = np.zeros((n_z, n_y, n_x_samp, n_files))
    q_auto_out = np.zeros((n_z, n_y, n_x_samp, n_files)) 
    t_auto_out = np.zeros((n_z, n_y, n_x_samp, n_files))  

    #Griffin -- double check I need these
    q_sed_fluxi_out = np.zeros((n_z, n_y, n_x_samp, n_files))
    q_sed_fluxc_out = np.zeros((n_z, n_y, n_x_samp, n_files))
    q_sed_flux_tot = np.zeros((n_z, n_y, n_x_samp, n_files))

    # Loop over files
    for ifile, file_time in enumerate(file_times):
        print(file_time)

        # Initialize
        zTin = np.zeros((n_z, n_y, n_x))
        zqin = np.zeros((n_z, n_y, n_x))
        zTout = np.zeros((n_z, n_y, n_x))
        zqout = np.zeros((n_z, n_y, n_x))
        tabs = np.zeros((n_z, n_y, n_x))
        t = np.zeros((n_z, n_y, n_x))
        qt = np.zeros((n_z, n_y, n_x))
        dqp = np.zeros((n_z, n_y, n_x))
        tflux_z = np.zeros((n_z, n_y, n_x))
        qtflux_z = np.zeros((n_z, n_y, n_x))
        qpflux_z = np.zeros((n_z, n_y, n_x))
        w = np.zeros((n_z, n_y, n_x))
        flux_down = np.zeros((n_y, n_x))
        flux_up = np.zeros((n_y, n_x))
        tfull_flux_diff_z = np.zeros((n_z, n_y, n_x))

        zT_adv_out = np.zeros((n_z, n_y, n_x))
        zq_adv_out = np.zeros((n_z, n_y, n_x))
        zt_auto_out = np.zeros((n_z, n_y, n_x))
        zq_auto_out = np.zeros((n_z, n_y, n_x))

        zq_sed_fluxi_out = np.zeros((n_z, n_y, n_x))
        zq_sed_fluxc_out = np.zeros((n_z, n_y, n_x))
        zq_sed_flux_tot = np.zeros((n_z, n_y, n_x))


        tflux_diff_z = np.zeros((n_z, n_y, n_x))
        qtflux_diff_z = np.zeros((n_z, n_y, n_x))
        tflux_diff_coarse_z = np.zeros((n_z, n_y, n_x))
        qtflux_diff_coarse_z = np.zeros((n_z, n_y, n_x))
        qpflux_diff_coarse_z = np.zeros((n_z, n_y, n_x))


        cloud_qt_flux = np.zeros(
            (n_z, n_y, n_x))  
        cloud_lat_heat_flux = np.zeros(
            (n_z, n_y, n_x))  

        zcos_lat = np.zeros((n_y, n_x))
        zsin_lon = np.zeros((n_y, n_x))
        zsfc_pres = np.zeros((n_y, n_x))
        zskt = np.zeros((n_y, n_x))
        zland_frac = np.zeros((n_z, n_y, n_x)) 

        # Variables to calculate the diffusivity.
        tkz_z = np.zeros((n_z, n_y, n_x))  
        Pr1 = np.zeros((n_z, n_y, n_x))  
        tkz_z_res = np.zeros((n_z, n_y, n_x))  
        Pr1_res = np.zeros((n_z, n_y, n_x))  
        tkh_z = np.zeros((n_z, n_y, n_x))  

        remainder = 10 - len(str(file_time))
        zeros = ''.join(['0']*remainder)
        filename_wildcard = base_dir + 'DYAMOND2_coars_9216x4608x74_10s_4608_*' + '*_' + zeros + str(file_time) + '.atm.3D_resolved.nc4'
        filename = glob.glob(filename_wildcard)
        print(filename[0])

        f = Dataset(filename[0], mode='r')
        
        tabs = f.variables['TABS_SIGMA'][:]  # absolute temperature (K)
        Qrad = f.variables['QRAD_SIGMA'][:] / 86400  # rad heating rate (K/s)
        qt = (f.variables['QV_SIGMA'][:] + f.variables['QC_SIGMA'][:] + f.variables['QI_SIGMA'][:]) / 1000.0  # total non-precip water (kg/kg)
        qp = f.variables['QP_SIGMA'][:] / 1000.0  # precipitating water (kg/kg)
        dqp = f.variables['QP_MICRO_SIGMA'][:] / 1000.0  # kg/kg/s - Taking coarse result since it is a smaller value to predict!
        qpflux_z_coarse = f.variables['RHOQPW_SIGMA'][:] / 1000.0  # SGS qp flux kg/m^2/s #I need it to the calculation of the dL/dz term
        tflux_z = f.variables['T_FLUX_Z_OUT_SUBGRID_SIGMA'][:]  # SGS t flux K kg/m^2/s - new name in new version
        qtflux_z = f.variables['Q_FLUX_Z_OUT_SUBGRID_SIGMA'][:] / 1000.0  # SGS qt flux kg/m^2/s
        qpflux_z = f.variables['QP_FLUX_Z_OUT_SUBGRID_SIGMA'][:] / 1000.0  # SGS qp flux kg/m^2/s
        
        if flag_dict['do_sedimentation']:
            cloud_qt_flux = f.variables['SED_SIGMA'][:] / 1000.0
            cloud_lat_heat_flux = f.variables['LSED_SIGMA'][:]  

        w = f.variables['W'][:]  # m/s
        precip = f.variables['PREC_SIGMA'][:]  # precipitation flux kg/m^2/s
        
        if flag_dict['do_qp_diff_corr_to_T']:
            qpflux_diff_coarse_z = f.variables['RHOQPS_SIGMA'][
                                   :] / 1000.0  # SGS qp flux kg/m^2/s Note that I need this variable

        f.close()

        zTin = tabs
        zqin = qt

        # approach where find tendency of hL without qp
        # use omp since heating as condensate changes to precipitation
        # of different phase also increases hL
        a_pr = 1.0 / (atmos_physics.tprmax - atmos_physics.tprmin)
        omp = np.maximum(0.0, np.minimum(1.0, (tabs - atmos_physics.tprmin) * a_pr))
        fac = (atmos_physics.L + atmos_physics.Lf * (1.0 - omp)) / atmos_physics.cp

        # follow simplified version of advect_scalar3D.f90 for vertical advection
        rho_dz = atmos_physics.vertical_diff(rho, z)

        if flag_dict['ver_adv_correct']:
            zT_adv_out = tflux_z
            zq_adv_out = qtflux_z

        if flag_dict['do_dqp']:
            zq_auto_out[:, :, :] = - dqp[:, :, :]

        zq_sed_fluxc_out = ((atmos_physics.L + atmos_physics.Lf) * cloud_qt_flux + cloud_lat_heat_flux) / atmos_physics.Lf
        zq_sed_fluxi_out = - (atmos_physics.L * cloud_qt_flux + cloud_lat_heat_flux) / atmos_physics.Lf
        zq_sed_flux_tot  = cloud_qt_flux


        # additional term from variation of latent heat of condensation
        # only account for variation in z here
        breakpoint()
        dfac_dz = np.zeros((n_z, n_y, n_x))
        for k in range(n_z - 1):
            kb = max(0, k - 1)
            dfac_dz[k, :, :] = (fac[k + 1, :, :] - fac[k, :, :]) / rho_dz[k] * rho[k]
        zTout = zTout + dfac_dz * (qpflux_z_coarse + qpflux_diff_coarse_z - precip) / rho[:, None, None]  
 
        if flag_dict['cos_lat']:
            zcos_lat[:, :] = np.cos(np.radians(y[:, None]))
            
        if flag_dict['sin_lon']:
            zsin_lon[:, :] = np.sin(np.radians(x[None, :]))
            
        if flag_dict['sfc_pres']:
            zsfc_pres[:, :] = SFC_PRES
            
        if flag_dict['skt']:
            zskt[:, :] = SKT
            
        if flag_dict['land_frac']:
            zland_frac[:, :, :] = terra

        # Loop over y
        #check if I can get rid of this -- I don't want to subsample x
        for j in range(zTin.shape[1]): 
            

            ind_x = np.arange(0,768,1)
            Tin[:, j, :, ifile] = zTin[0:n_z_input, j, :][:, ind_x]
            
            if flag_dict['land_frac']:
                land_frac[:, j, :, ifile] = zland_frac[0:flag_dict['land_frac_level'], j, :][:, ind_x]
            
            qin[:, j, :, ifile] = zqin.squeeze()[0:n_z_input, j, :][:, ind_x]
            Tout[:, j, :, ifile] = zTout[:, j, :][:, ind_x]

            T_adv_out[:, j, :, ifile] = zT_adv_out[:, j, :][:, ind_x]
            q_adv_out[:, j, :, ifile] = zq_adv_out[:, j, :][:, ind_x]
            q_auto_out[:, j, :, ifile] = zq_auto_out[:, j, :][:, ind_x]
            q_sed_fluxi_out[:, j, :, ifile] = zq_sed_fluxi_out[:, j, :][:, ind_x]
            q_sed_fluxc_out[:, j, :, ifile] = zq_sed_fluxc_out[:, j, :][:, ind_x]
            q_sed_flux_tot[:, j, :, ifile] = zq_sed_flux_tot[:, j, :][:, ind_x]

            t_auto_out[:, j, :, ifile] = zt_auto_out[:, j, :][:, ind_x]
  
            if flag_dict['cos_lat']:
                cos_lat[j, :, ifile] = zcos_lat[j, :][ind_x]
            if flag_dict['sin_lon']:
                sin_lon[j, :, ifile] = zsin_lon[j, :][ind_x]
            if flag_dict['sfc_pres']:
                sfc_pres[j, :, ifile] = zsfc_pres[j, :][ind_x]
            if flag_dict['skt']:
                skt[j, :, ifile] = zskt[j, :][ind_x]

    # Reshape array to be n_z n_y n_samp*n_file
    print('a', Tin.shape)
    Tin = np.moveaxis(Tin, 2, 3)
    print('b', Tin.shape)
    qin = np.moveaxis(qin, 2, 3)
    Tout = np.moveaxis(Tout, 2, 3)

    T_adv_out = np.moveaxis(T_adv_out, 2, 3)
    q_adv_out = np.moveaxis(q_adv_out, 2, 3)
    q_auto_out = np.moveaxis(q_auto_out, 2, 3)
    q_sed_flux_tot = np.moveaxis(q_sed_flux_tot, 2, 3)
    q_sed_fluxi_out = np.moveaxis(q_sed_fluxi_out, 2, 3)
    q_sed_fluxc_out = np.moveaxis(q_sed_fluxc_out, 2, 3)

    Tin = np.reshape(Tin, (n_z_input, n_y, -1))
    print('c', Tin.shape)
    qin = np.reshape(qin, (n_z_input, n_y, -1))
    Tout = np.reshape(Tout, (n_z, n_y, -1))

    T_adv_out = np.reshape(T_adv_out, (n_z, n_y, -1))
    q_adv_out = np.reshape(q_adv_out, (n_z, n_y, -1))
    q_auto_out = np.reshape(q_auto_out, (n_z, n_y, -1))
    q_sed_fluxi_out = np.reshape(q_sed_fluxi_out, (n_z, n_y, -1))
    q_sed_fluxc_out = np.reshape(q_sed_fluxc_out, (n_z, n_y, -1))
    q_sed_flux_tot = np.reshape(q_sed_flux_tot, (n_z, n_y, -1))

    t_auto_out = np.reshape(t_auto_out, (n_z, n_y, -1))

    if flag_dict['cos_lat']:
        cos_lat = np.expand_dims(cos_lat, axis=0)
        cos_lat = np.moveaxis(cos_lat, 2, 3)
        cos_lat = np.reshape(cos_lat, (1, n_y, -1))
    if flag_dict['sin_lon']:
        sin_lon = np.expand_dims(sin_lon, axis=0)
        sin_lon = np.moveaxis(sin_lon, 2, 3)
        sin_lon = np.reshape(sin_lon, (1, n_y, -1))
    if flag_dict['sfc_pres']:
        sfc_pres = np.expand_dims(sfc_pres, axis=0)
        sfc_pres = np.moveaxis(sfc_pres, 2, 3)
        sfc_pres = np.reshape(sfc_pres, (1, n_y, -1))
    if flag_dict['skt']:
        skt = np.expand_dims(skt, axis=0)
        skt = np.moveaxis(skt, 2, 3)
        skt = np.reshape(skt, (1, n_y, -1))
    if flag_dict['land_frac']:
        land_frac = np.moveaxis(land_frac, 2, 3)
        land_frac = np.reshape(land_frac, (n_z_input, n_y, -1))

    #Separate out training and testing files
    i70 = int(train_size * Tin.shape[2])
    randind_trn = np.random.permutation(i70)
    tst_list = np.arange(i70, int(Tin.shape[2]), 1)
    randind_tst = np.random.permutation(tst_list)

    data_specific_description = create_specific_data_string_desc(flag_dict)

    if rewight_outputs: 
            norm_list = calculate_renormalization_factors(Tout[0:flag_dict['input_upper_lev'],:, randind_trn],
                                                          T_adv_out[0:flag_dict['input_upper_lev'],:, randind_trn],
                                                          q_adv_out[0:flag_dict['input_upper_lev'],:, randind_trn],
                                                          q_auto_out[0:flag_dict['input_upper_lev'],:, randind_trn],
                                                          q_sed_flux_tot[0:flag_dict['input_upper_lev'],:, randind_trn],
                                                          rho_dz)       
    #Griffin -- I want to change to a dictionary
    train_input_list = []
    test_input_list = []
    name_list = ''


    # Choosing the lists that will be dumped in the pickle file...
    print('d', Tin.shape)
    print("Tin", Tin.shape)
    print("Tin Train",Tin[:, :, randind_trn].shape)
    if flag_dict['Tin_feature']:
        print("Tin_feature")
        train_input_list.append(np.float32(Tin[:, :, randind_trn]))
        test_input_list.append(np.float32(Tin[:, :, randind_tst]))
        name_list = name_list + '_T_'
    if flag_dict['qin_feature']:
        print("The length of the list is:", len(train_input_list))
        train_input_list.append(np.float32(qin[:, :, randind_trn]))
        test_input_list.append(np.float32(qin[:, :, randind_tst]))
        name_list = name_list + '_Q_'
    if flag_dict['cos_lat']:
        print('cos_lat')
        cos_lat = np.squeeze(cos_lat)
        train_input_list.append(np.float32(cos_lat[:, randind_trn]))
        test_input_list.append(np.float32(cos_lat[:, randind_tst]))
        name_list = name_list + '_CosLT_'
    if flag_dict['sin_lon']:
        print('sin_lon')
        sin_lon = np.squeeze(sin_lon)
        train_input_list.append(np.float32(sin_lon[:, randind_trn]))
        test_input_list.append(np.float32(sin_lon[:, randind_tst]))
        name_list = name_list + '_SinLN_'
    if flag_dict['sfc_pres']:
        print('sfc_pres')
        sfc_pres = np.squeeze(sfc_pres)
        train_input_list.append(np.float32(sfc_pres[:, randind_trn]))
        test_input_list.append(np.float32(sfc_pres[:, randind_tst]))
        name_list = name_list + '_Ps_'
    if flag_dict['skt']:
        print('skt')
        skt = np.squeeze(skt)
        train_input_list.append(np.float32(skt[:, randind_trn]))
        test_input_list.append(np.float32(skt[:, randind_tst]))
        name_list = name_list + '_SKT_'
    if flag_dict['land_frac']:
        print('land_frac')
        train_input_list.append(np.float32(land_frac[:, :, randind_trn]))
        test_input_list.append(np.float32(land_frac[:, :, randind_tst]))
        name_list = name_list + '_LF_'

    if flag_dict['predict_tendencies']:
        print('Tout')
        train_input_list.append(np.float32(Tout[0:flag_dict['input_upper_lev'],:, randind_trn]))
        test_input_list.append(np.float32(Tout[0:flag_dict['input_upper_lev'],:, randind_tst]))
        print('Tadv')
        #Recent Griffin Comment -- change 1 below to 0 based on POG comment that bottom level is not zero
        train_input_list.extend([np.float32(T_adv_out[0:flag_dict['input_upper_lev'], :, randind_trn]), np.float32(q_adv_out[0:flag_dict['input_upper_lev'], :, randind_trn])])
        test_input_list.extend([np.float32(T_adv_out[0:flag_dict['input_upper_lev'], :, randind_tst]), np.float32(q_adv_out[0:flag_dict['input_upper_lev'], :, randind_tst])])

        do_sedimentation1 = 1
        do_autoconversion1 = 1
        if do_autoconversion1 :
            print('Qout')
            train_input_list.append(np.float32(q_auto_out[0:flag_dict['input_upper_lev'], :, randind_trn]))
            test_input_list.append(np.float32(q_auto_out[0:flag_dict['input_upper_lev'], :, randind_tst]))
            print('Qsed')
            train_input_list.append(np.float32(q_sed_flux_tot[0:flag_dict['input_upper_lev'], :, randind_trn]))
            test_input_list.append(np.float32(q_sed_flux_tot[0:flag_dict['input_upper_lev'], :, randind_tst]))

    train_input_list.extend([y, z, p, rho])
    test_input_list.extend([y, z, p, rho])
    
    if flag_dict['no_poles']:
        lat_lower_bound = np.where(y < -70)[0][-1] + 1
        lat_upper_bound = np.where(y > 70)[0][0] 
        for i in range(len(train_input_list)):
            obj_temp = train_input_list[i]
            dims_now = np.asarray(obj_temp.shape)
            if len(y) in dims_now:
                index = np.where(dims_now == len(y))[0][0]
                new_array = obj_temp.take(indices=np.arange(lat_lower_bound,lat_upper_bound,1), axis=index)
                train_input_list[i] = new_array
        name_list = "NP_" + name_list 
        
    if rewight_outputs:
        train_input_list.append(norm_list)
        test_input_list.append(norm_list)
            
    print("The length of the list is:", len(train_input_list))
    name_list = str(flag_dict['unique_name']) + name_list + "L"+str(len(train_input_list[0]))+'_'
    pickle.dump(train_input_list,open(output_dir + name_list + 'training.pkl', 'wb'))
    pickle.dump(test_input_list,open(output_dir + name_list + 'testing.pkl', 'wb'))
    print("completed")
