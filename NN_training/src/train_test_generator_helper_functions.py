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
import dask.array as da


def write_netcdf_rf(est_str, datasource, output_vert_vars, output_vert_dim, rain_only=False,
                    no_cos=False, use_rh=False,scale_per_column=False,
                    rewight_outputs=False,weight_list=[1,1],is_cheyenne=False,exclusion_flag=False,ind1_exc=0,ind2_exc=0):
    # Set output filename
    if is_cheyenne == False: #On aimsir/esker
        base_dir = '/net/aimsir/archive1/janniy/'
    else:
        base_dir = '/glade/scratch/janniy/'
    if exclusion_flag:
        output_filename = base_dir + 'mldata_tmp/gcm_regressors/' + est_str + str(ind1_exc) + str(ind2_exc)+ '.nc'
    else:
        output_filename = base_dir + 'mldata_tmp/gcm_regressors/'+est_str+'.nc'
    if exclusion_flag:
        est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho = \
            pickle.load(open(base_dir + 'mldata_tmp/regressors/' + est_str  + str(ind1_exc) + str(ind2_exc)+ '.pkl', 'rb'))
    else:
        est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho = \
            pickle.load(open(base_dir + 'mldata_tmp/regressors/' + est_str + '.pkl', 'rb'))

    # determine the maximum number of nodes and the number of features/outputs
    estimators = est.estimators_
    n_trees = len(estimators)
    n_nodes = np.zeros(n_trees, dtype=np.int32)
    for itree in range(n_trees):
        tree = estimators[itree].tree_
        n_nodes[itree] = tree.node_count
    max_n_nodes = np.amax(n_nodes)
    print("Maximum number of nodes across trees:")
    print(max_n_nodes)
    print("Average number of nodes across trees:")
    print(np.mean(n_nodes))
    n_features = estimators[0].tree_.n_features
    n_outputs = estimators[0].tree_.n_outputs

    # populate arrays that describe trees
    children_left = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    children_right = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    split_feature = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    n_node_samples = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    threshold = np.zeros((max_n_nodes, n_trees), dtype=np.float32)
    values_predicted = np.zeros((n_outputs, max_n_nodes, n_trees), dtype=np.float32)  # Yani modified to reduce spave

    # note for python, slices don't include upper index!
    # inverse transform outputs here to speed up the GCM parameterization
    n_leaf_nodes = 0
    n_samples_leaf_nodes = 0
    for itree in range(n_trees):
        tree = estimators[itree].tree_
        children_left[:n_nodes[itree], itree] = tree.children_left
        children_right[:n_nodes[itree], itree] = tree.children_right
        split_feature[:n_nodes[itree], itree] = tree.feature
        threshold[:n_nodes[itree], itree] = tree.threshold
        n_node_samples[:n_nodes[itree], itree] = tree.n_node_samples
        for inode in range(n_nodes[itree]):
            # values_predicted[:,inode,itree] = np.float32(ml_load.inverse_transform_data(o_ppi, o_pp, (tree.value[inode,:]).T, z))  # Yani modified to reduce spave (float32)
            o_dict = ml_load.unpack_list((tree.value[inode, :]).T, output_vert_vars, output_vert_dim)
            values_predicted[:, inode, itree] = np.float32(
                ml_load.inverse_transform_data_generalized(o_ppi, o_pp, o_dict, output_vert_vars, z, scale_per_column,
                                                           rewight_outputs=rewight_outputs,
                                                           weight_list=weight_list))  # Makes sure we get our outputs in the correct units.

            if children_left[inode, itree] == children_right[inode, itree]:  # leaf node
                n_leaf_nodes = n_leaf_nodes + 1
                n_samples_leaf_nodes = n_samples_leaf_nodes + n_node_samples[inode, itree]

    print("Average number of leaf nodes across trees:")
    print(n_leaf_nodes / n_trees)

    # chance of not being included in bootstrap sample is (1-1/n)^n
    # which is 1/e for large n
    # note each tree has only about (1-1/e)*n_trn_exs due to bagging
    # which is 63% of them
    # only seem to keep one when there are non-unique samples
    print("Average number of samples per leaf node:")
    print(n_samples_leaf_nodes / n_leaf_nodes)

    if f_ppi['name'] != 'NoScaler':
        raise ValueError('Incorrect scaler name - Cannot treat any other case - in RF no need to')

    # Write to file
    ncfile = Dataset(output_filename, 'w', format="NETCDF3_CLASSIC")
    # Write the dimensions
    ncfile.createDimension('dim_nodes', max_n_nodes)
    ncfile.createDimension('dim_trees', n_trees)
    ncfile.createDimension('dim_features', n_features)
    ncfile.createDimension('dim_outputs', n_outputs)

    # Create variable entries in the file
    nc_n_nodes = ncfile.createVariable('n_nodes', np.dtype('int32').char, ('dim_trees'))
    nc_children_left = ncfile.createVariable('children_left', np.dtype('int32').char, ('dim_nodes', 'dim_trees'))
    nc_children_right = ncfile.createVariable('children_right', np.dtype('int32').char, ('dim_nodes', 'dim_trees'))
    nc_split_feature = ncfile.createVariable('split_feature', np.dtype('int32').char, ('dim_nodes', 'dim_trees'))
    nc_threshold = ncfile.createVariable('threshold', np.dtype('float32').char, ('dim_nodes', 'dim_trees'))
    nc_values_predicted = ncfile.createVariable('values_predicted', np.dtype('float32').char,
                                                ('dim_outputs', 'dim_nodes', 'dim_trees'))

    # Write variables and close file
    nc_n_nodes[:] = n_nodes
    nc_children_left[:] = children_left
    nc_children_right[:] = children_right
    nc_split_feature[:] = split_feature
    nc_threshold[:] = threshold
    nc_values_predicted[:] = np.float32(values_predicted)

    # Write global file attributes
    ncfile.description = est_str
    ncfile.close()


def write_netcdf_nn(est_str, datasource, rain_only=False, no_cos=False, use_rh=False, is_cheyenne=False):
    # Set output filename
    if is_cheyenne == False:  # On aimsir/esker
        base_dir = '/net/aimsir/archive1/janniy/'
    else:
        base_dir = '/glade/work/janniy/'

    output_filename = base_dir + 'mldata/gcm_regressors/' + est_str + '.nc'
    # Load rf and preprocessors
    est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho = \
        pickle.load(open(base_dir + 'mldata/regressors/' + est_str + '.pkl', 'rb'))
    # Need to transform some data for preprocessors to be able to export params
    f, o, _, _, _, _, = ml_load.LoadData(datasource,
                                         max_z=max(z),
                                         rain_only=rain_only,
                                         no_cos=no_cos,
                                         use_rh=use_rh)
    f_scl = ml_load.transform_data(f_ppi, f_pp, f, z)
    _ = ml_load.transform_data(o_ppi, o_pp, o, z)
    # Also need to use the predict method to be able to export ANN params
    _ = est.predict(f_scl)

    # Grab weights
    w1 = est.get_parameters()[0].weights
    w2 = est.get_parameters()[1].weights
    b1 = est.get_parameters()[0].biases
    b2 = est.get_parameters()[1].biases

    # Grab input and output normalization
    if f_ppi['name'] == 'StandardScaler':
        fscale_mean = f_pp.mean_
        fscale_stnd = f_pp.scale_
    else:
        raise ValueError('Incorrect scaler name')

    if o_ppi['name'] == 'SimpleO':
        Nlev = len(z)
        oscale = np.zeros(b2.shape)
        oscale[:Nlev] = 1.0 / o_pp[0]
        oscale[Nlev:] = 1.0 / o_pp[1]
    elif o_ppi['name'] == 'StandardScaler':
        oscale_mean = o_pp.mean_
        oscale_stnd = o_pp.scale_
    else:
        raise ValueError('Incorrect scaler name')

        # Write weights to file
    ncfile = Dataset(output_filename, 'w', format="NETCDF3_CLASSIC")
    # Write the dimensions
    ncfile.createDimension('N_in', w1.shape[0])
    ncfile.createDimension('N_h1', w1.shape[1])
    ncfile.createDimension('N_out', w2.shape[1])
    # Create variable entries in the file
    nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                  ('N_h1', 'N_in'))  # Reverse dims
    nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                  ('N_out', 'N_h1'))
    nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                  ('N_h1'))
    nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                  ('N_out'))
    nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                           np.dtype('float32').char, ('N_in'))
    nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                           np.dtype('float32').char, ('N_in'))
    if o_ppi['name'] == 'SimpleO':
        nc_oscale = ncfile.createVariable('oscale',
                                          np.dtype('float32').char,
                                          ('N_out'))
    else:
        nc_oscale_mean = ncfile.createVariable('oscale_mean', np.dtype('float32').char, ('N_out'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd', np.dtype('float32').char, ('N_out'))

    # Write variables and close file - transpose because fortran reads it in
    # "backwards"
    nc_w1[:] = w1.T
    nc_w2[:] = w2.T
    nc_b1[:] = b1
    nc_b2[:] = b2
    nc_fscale_mean[:] = fscale_mean
    nc_fscale_stnd[:] = fscale_stnd

    if o_ppi['name'] == 'SimpleO':
        nc_oscale[:] = oscale
    else:
        nc_oscale_mean[:] = oscale_mean
        nc_oscale_stnd[:] = oscale_stnd

    # Write global file attributes
    ncfile.description = est_str
    ncfile.close()


def create_z_grad_plus_surf_var(variable):
    T_grad_in = np.zeros(variable.shape)
    T_grad_in[0, :, :] = variable[0, :, :]  # The surface temperature
    T_grad_in[1:variable.shape[0], :, :] = variable[1:variable.shape[0], :, :] - variable[0:variable.shape[0] - 1, :, :]
    return T_grad_in


def create_difference_from_surface(variable):
    T_s_duff = np.zeros(variable.shape)
    T_s_duff[0, :, :] = variable[0, :, :]  # The surface temperature
    for ind in range(variable.shape[0] - 1):
        print(ind)
        T_s_duff[ind + 1, :, :] = variable[0, :, :] - variable[ind + 1, :, :]
    return T_s_duff

def vertical_smooth(variable):
    var_vert_avg = np.zeros(variable.shape)
    var_vert_avg[:-1, :, :] =0.5 *( variable[:-1, :, :] + variable[1:, :, :])
    var_vert_avg[-1,:,:] = variable[-1, :, :]
    return var_vert_avg

def create_specific_data_string_desc(flag_dict):
    """create the name of the output files according to input flags
    Args:
     flag_dict (dict): including the specific configuration that we want to calculate the outputs for
    """

    data_specific_description = str(flag_dict['Tin_feature'])[0] +  \
                                str(flag_dict['qin_feature'])[0] + \
                                str(flag_dict['predict_tendencies'])[0] + \
                                str(flag_dict['land_frac'])[0] + \
                                str(flag_dict['skt'])[0] + \
                                str(flag_dict['sfc_pres'])[0] + \
                                str(flag_dict['cos_lat'])[0] + \
                                str(flag_dict['sin_lon'])[0] + \
                                str(flag_dict['no_poles'])[0]
    
    return data_specific_description


def print_simulation_decription(filename):
    i = 4
    print('do_dqp=', filename[i])
    i = i + 1
    print('ver_adv_correct=', filename[i])
    i = i + 1
    print('do_hor_wind_input=', filename[i])
    i = i + 1
    print('do_ver_wind_input=', filename[i])
    i = i + 1
    print('do_z_diffusion=', filename[i])
    i = i + 1
    print('do_q_T_surf_fluxes=', filename[i])
    i = i + 1
    print('do_surf_wind=', filename[i])
    i = i + 1
    print('do_sedimentation=', filename[i])
    i = i + 1
    print('do_radiation_output=', filename[i])
    i = i + 1
    print('rad_level=', filename[i:i + 2])
    i = i + 2
    print('do_qp_as_var=', filename[i])
    i = i + 1
    print('do_fall_tend=', filename[i])
    i = i + 1
    print('Tin_feature=', filename[i])
    i = i + 1
    print('Tin_z_grad_feature=', filename[i])
    i = i + 1
    print('qin_feature=', filename[i])
    i = i + 1
    print('qin_z_grad_feature=', filename[i])
    i = i + 1
    print('input_upper_lev=', filename[i:i + 2])
    i = i + 2
    print('predict_tendencies=', filename[i])
    i = i + 1
    print('do_qp_diff_corr_to_T=', filename[i])
    i = i + 1
    print('do_q_T_surf_fluxes_correction=', filename[i])
    i = i + 1
    print('do_t_strat_correction=', filename[i])
    i = i + 1
    print('output_precip=', filename[i])
    i = i + 1
    print('do_radiation_in_Tz', filename[i])
    i = i + 1
    print('do_z_diffusion_correction', filename[i])
    i = i + 1
    print('calc_tkz_z=', filename[i])
    i = i + 1
    print('calc_tkz_z_correction=', filename[i])
    i = i + 1
    print('resolution=', filename[i:i + 2])
    i = i + 2
    print('tkz_levels=', filename[i:i + 2])
    i = i + 1
    print('Tin_s_diff_feature=', filename[i])
    i = i + 1
    print('qin_s_diff_feature=', filename[i])
    i = i + 1
    print['dist_From_eq_in=', filename[i]]
    i = i + 1
    print['T_instead_of_Tabs=', filename[i]]
    i = i + 1
    print['tabs_resolved_init=', filename[i]]
    i = i + 1
    print['qn_coarse_init=', filename[i]]
    i = i + 1
    print['qn_resolved_as_var=', filename[i]]
    i = i + 1
    print['strat_corr_level=', filename[i]]
    i = i + 2
    print['sed_level=', filename[i]]


def calculate_renormalization_factors(Tout, tflux_z, qtflux_z, qmic, qsed, rho_dz):
    '''I need to renormalize somehow the outputs.
     here I assume I am using a flux form, renormalizing (T_rad+rest , Tadv, qadv, qmic, qsed):'''

    n_z = tflux_z.shape[0]
    zTout = np.zeros(tflux_z.shape)
    zqout = np.zeros(qtflux_z.shape)
    zqsed = np.zeros(qsed.shape)
    #Flux to tendency:
    for k in range(n_z - 1):
        zTout[k, :, :] = -(tflux_z[k + 1, :, :] - tflux_z[k, :, :]) / rho_dz[k]
        zqout[k, :, :] = -(qtflux_z[k + 1, :, :] - qtflux_z[k, :, :]) / rho_dz[k]
        zqsed[k, :, :] = -(qsed[k + 1, :, :] - qsed[k, :, :]) / rho_dz[k]

    # flux is defined to be zero at top half-level
    zTout[n_z - 1, :, :] = -(0.0 - tflux_z[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqout[n_z - 1, :, :] = -(0.0 - qtflux_z[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqsed[k, :, :] = -(0.0 - qsed[n_z - 1, :, :]) / rho_dz[n_z -1]

    #Rescale humudity tendencies
    zqsed = zqsed * atmos_physics.L / atmos_physics.cp
    qmic = qmic * atmos_physics.L / atmos_physics.cp
    zqout = zqout * atmos_physics.L / atmos_physics.cp

    std1 = np.std(Tout)
    std2 = np.std(zTout)
    std3 = np.std(zqout)
    std4 = np.std(qmic)
    std5 = np.std(zqsed)

    std_min = min(std1,std2,std3,std4,std5)
    std1 = std1/std_min
    std2 = std2 / std_min
    std3 = std3 / std_min
    std4 = std4 / std_min
    std5 = std5 / std_min
    return np.array([std1,std2,std3,std4,std5])


def calculate_renormalization_factors_sample(Tout, tflux_z, qtflux_z, qmic, qsed, rho_dz):
    '''I need to renormalize somehow the outputs.
     here I assume I am using a flux form, renormalizing (T_rad+rest , Tadv, qadv, qmic, qsed):'''

    n_z = tflux_z.shape[0]
    zTout = np.zeros(tflux_z.shape)
    zqout = np.zeros(qtflux_z.shape)
    zqsed = np.zeros(qsed.shape)
    #Flux to tendency:
    for k in range(n_z - 1):
        zTout[k, :] = -(tflux_z[k + 1, :] - tflux_z[k, :]) / rho_dz[k]
        zqout[k, :] = -(qtflux_z[k + 1, :] - qtflux_z[k, :]) / rho_dz[k]
        zqsed[k, :] = -(qsed[k + 1, :] - qsed[k, :]) / rho_dz[k]

    # flux is defined to be zero at top half-level
    zTout[n_z - 1, :] = -(0.0 - tflux_z[n_z - 1, :]) / rho_dz[n_z - 1]
    zqout[n_z - 1, :] = -(0.0 - qtflux_z[n_z - 1, :]) / rho_dz[n_z - 1]
    zqsed[k, :] = -(0.0 - qsed[n_z - 1, :]) / rho_dz[n_z -1]

    #Rescale humudity tendencies
    zqsed = zqsed * atmos_physics.L / atmos_physics.cp
    qmic = qmic * atmos_physics.L / atmos_physics.cp
    zqout = zqout * atmos_physics.L / atmos_physics.cp

    std1 = np.std(Tout)
    std2 = np.std(zTout)
    std3 = np.std(zqout)
    std4 = np.std(qmic)
    std5 = np.std(zqsed)

    std_min = min(std1,std2,std3,std4,std5)
    std1 = std1/std_min
    std2 = std2 / std_min
    std3 = std3 / std_min
    std4 = std4 / std_min
    std5 = std5 / std_min
    return np.array([std1,std2,std3,std4,std5])

def calculate_diffusion_renormalization_factors():
    '''This is normalizing the diffusivity and surface fluxes'''
    print('rescaling with 2.5 2.5 1 the surface flux')
    std1 = 2.5 #Choosing larger values for the surface fluxes
    std2 = 2.5
    std3 = 1.0
    return [std1, std2, std3]



def calculate_renormalization_diff_tend(tflux_diff_z, qtflux_diff_z, rho_dz):
    '''I need to renormalize somehow the outputs.
     here I assume I am using a flux form, renormalizing (T_rad+rest , Tadv, qadv, qmic, qsed):'''

    n_z = tflux_diff_z.shape[0]
    zTout = np.zeros(tflux_diff_z.shape)
    zqout = np.zeros(qtflux_diff_z.shape)
    #Flux to tendency:
    for k in range(n_z - 1):
        zTout[k, :, :] = -(tflux_diff_z[k + 1, :, :] - tflux_diff_z[k, :, :]) / rho_dz[k]
        zqout[k, :, :] = -(qtflux_diff_z[k + 1, :, :] - qtflux_diff_z[k, :, :]) / rho_dz[k]

    # flux is defined to be zero at top half-level
    zTout[n_z - 1, :, :] = -(0.0 - tflux_diff_z[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqout[n_z - 1, :, :] = -(0.0 - qtflux_diff_z[n_z - 1, :, :]) / rho_dz[n_z - 1]

    #Rescale humudity tendencies
    zqout = zqout * atmos_physics.L / atmos_physics.cp

    std1 = np.std(zTout)
    std2 = np.std(zqout)

    std_min = min(std1,std2)
    std1 = std1/std_min
    std2 = std2 / std_min
    return [std1,std2]

def calculate_renormalization_diff_tend_separate_flux(tflux_diff_z, qtflux_diff_z, tsurf, qsurf, rho_dz):

    n_z = tflux_diff_z.shape[0]
    zTout = np.zeros(tflux_diff_z.shape)
    zqout = np.zeros(qtflux_diff_z.shape)
    #Flux to tendency:
    for k in range(n_z - 1):
        zTout[k, :, :] = -(tflux_diff_z[k + 1, :, :] - tflux_diff_z[k, :, :]) / rho_dz[k+1] #The rho_dz[k+1] is because how we plug in the inputs they have different indexing.
        zqout[k, :, :] = -(qtflux_diff_z[k + 1, :, :] - qtflux_diff_z[k, :, :]) / rho_dz[k+1]

    # flux is defined to be zero at top half-level
    zTout[n_z - 1, :, :] = -(0.0 - tflux_diff_z[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqout[n_z - 1, :, :] = -(0.0 - qtflux_diff_z[n_z - 1, :, :]) / rho_dz[n_z - 1]

    #Rescale humudity tendencies
    zqout = zqout * atmos_physics.L / atmos_physics.cp
    qsurf_scaled = qsurf * atmos_physics.L / atmos_physics.cp
    std1 = np.std(zTout)
    std2 = np.std(zqout)
    std3 = np.std(tsurf/ rho_dz[0])
    std4 = np.std(qsurf_scaled/ rho_dz[0])

    std_min = min(std1,std2,std3,std4)
    std1 = std1/std_min
    std2 = std2 / std_min
    std3 = std3 / std_min
    std4 = std4 / std_min
    return [std1,std2,std3,std4]

def calculate_renormalization_factors_all_diff(Tout, tflux_z, qtflux_z, qmic, qsed, Tdiff, qdiff, rho_dz):
    '''I need to renormalize somehow the outputs.
     here I assume I am using a flux form, renormalizing (T_rad+rest , Tadv, qadv, qmic, qsed):'''

    n_z = tflux_z.shape[0]
    zTout = np.zeros(tflux_z.shape)
    zqout = np.zeros(qtflux_z.shape)
    zqsed = np.zeros(qsed.shape)
    zTdiff = np.zeros(Tdiff.shape)
    zqdiff = np.zeros(qdiff.shape)

    #Flux to tendency:
    for k in range(n_z - 1):
        zTout[k, :, :] = -(tflux_z[k + 1, :, :] - tflux_z[k, :, :]) / rho_dz[k]
        zqout[k, :, :] = -(qtflux_z[k + 1, :, :] - qtflux_z[k, :, :]) / rho_dz[k]
        zqsed[k, :, :] = -(qsed[k + 1, :, :] - qsed[k, :, :]) / rho_dz[k]
        zTdiff[k, :, :] = -(Tdiff[k + 1, :, :] - Tdiff[k, :, :]) / rho_dz[k]
        zqdiff[k, :, :] = -(qdiff[k + 1, :, :] - qdiff[k, :, :]) / rho_dz[k]

    # flux is defined to be zero at top half-level
    zTout[n_z - 1, :, :] = -(0.0 - tflux_z[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqout[n_z - 1, :, :] = -(0.0 - qtflux_z[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqsed[k, :, :] = -(0.0 - qsed[n_z - 1, :, :]) / rho_dz[n_z -1]

    zTdiff[n_z - 1, :, :] = -(0.0 - Tdiff[n_z - 1, :, :]) / rho_dz[n_z - 1]
    zqdiff[n_z - 1, :, :] = -(0.0 - qdiff[n_z - 1, :, :]) / rho_dz[n_z - 1]

    #Rescale humudity tendencies
    zqsed = zqsed * atmos_physics.L / atmos_physics.cp
    qmic = qmic * atmos_physics.L / atmos_physics.cp
    zqout = zqout * atmos_physics.L / atmos_physics.cp
    zqdiff = zqdiff* atmos_physics.L / atmos_physics.cp

    std1 = np.std(Tout)
    std2 = np.std(zTout)
    std3 = np.std(zqout)
    std4 = np.std(qmic)
    std5 = np.std(zqsed)
    std6 = np.std(zTdiff)
    std7 = np.std(zqdiff)

    std_min = min(std1,std2,std3,std4,std5)
    std1 = std1/std_min
    std2 = std2 / std_min
    std3 = std3 / std_min
    std4 = std4 / std_min
    std5 = std5 / std_min
    std6 = std6 / std_min
    std7 = std7 / std_min
    return [std1,std2,std3,std4,std5,std6,std7]


def get_train_test_split(training_split, longitudes, times):
    pure_split = int(longitudes*times*training_split)
    return math.floor(float(pure_split) / longitudes) * longitudes


def calculate_renormalization_factors_mem(Tout, tflux_z, qtflux_z, qmic, qsed, rho_dz):
    '''Renormalize outputs assuming flux form renormalization for T_rad+rest, Tadv, qadv, qmic, qsed.'''

    # Calculate the differences along the vertical axis
    zTout = -(da.diff(tflux_z, axis=0) / rho_dz[:, None, None])
    zqout = -(da.diff(qtflux_z, axis=0) / rho_dz[:, None, None])
    zqsed = -(da.diff(qsed, axis=0) / rho_dz[:, None, None])

    # Handle the top boundary condition where the flux is defined to be zero at the top half-level
    zTout = da.concatenate([zTout, -tflux_z[-1, :, :][None, :, :] / rho_dz[-1]], axis=0)
    zqout = da.concatenate([zqout, -qtflux_z[-1, :, :][None, :, :] / rho_dz[-1]], axis=0)
    zqsed = da.concatenate([zqsed, -qsed[-1, :, :][None, :, :] / rho_dz[-1]], axis=0)

    # Rescale humidity tendencies
    L_cp_ratio = atmos_physics.L / atmos_physics.cp
    zqsed = zqsed * L_cp_ratio
    qmic = qmic * L_cp_ratio
    zqout = zqout * L_cp_ratio

    # Compute standard deviations
    std1 = da.std(Tout, axis=(0,1, 2))
    std2 = da.std(zTout, axis=(0,1, 2))
    std3 = da.std(zqout, axis=(0,1, 2))
    std4 = da.std(qmic, axis=(0,1, 2))
    std5 = da.std(zqsed, axis=(0,1, 2))

    # Stack the standard deviations and find the minimum
    std_stack = da.stack([std1, std2, std3, std4, std5])
    std_min = std_stack.min(axis=0)

    # Normalize the standard deviations
    std_factors = std_stack / std_min

    return std_factors.compute() 