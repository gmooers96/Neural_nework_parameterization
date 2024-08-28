import pdb 
import coarse_functions as cfunc
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from netCDF4 import Dataset
import time
import math
import xarray as xr
import glob
from os.path import exists
from scipy.interpolate import interp1d

import advect_scalar3D_f2py_my_min2

# 2D high-res output from Marat
ncfile = '/ocean/projects/ees220005p/janniy/gsam_data/SAM7.7_test/DYAMOND2_9216x4608x74_10s_4608_20200214121500_0000220410.2D_atm.nc'
f_solin = Dataset(ncfile, mode='r')
topog_high = f_solin.variables['ZSFC'][:]  # m
lat_high = f_solin.variables['lat'][:]  # m
lon_high = f_solin.variables['lon'][:]  # m
f_solin.close()

## Griffin's comment -- was originally 1
number_of_files = 10
## Griffin's comment - was originally 1080
start_time=137880       
interval = 360 
end_time = start_time  + interval * (number_of_files)


compare_flag = False #compare to matlab
test_mode = False #compare to matlab when writing NETCDF
test_mode1D = False #compare 1D to matlab when writing NETCDF - should be off.
res = [12]
dtn = 10



experiment = 'DYAMOND2_coars_9216x4608x74_10s_4608'
loaddir = '/ocean/projects/ees220005p/janniy/gsam_data/'

if test_mode or compare_flag:
    #TO DO - will have to read the 3D hi-res files (or a chunk of them), and calculate outputs... Could do on part of the file...
    print('To do test mode, for now I didnt write schemes')
    # filename = '/glade/scratch/janniy/ML_convection_data/qobs/qobskm12x576_576x1440x48_ctl_288_0001241100_0001_diff_coarse_space_corrected_tkz8.nc4'
else:
    filename = 'Dummy'

savedir = '/ocean/projects/ees220005p/gmooers/GM_Data/'
file_coarse1 = '/ocean/projects/ees220005p/janniy/gsam_data/DYAMOND2_coars_9216x4608x74_10s_4608_20200124120000_0000038880.atm.3DC.nc'
#Read hires
start = time.time()
f = Dataset(file_coarse1, mode='r')
lon = f.variables['lon'][:]
lat = f.variables['lat'][:]
z = f.variables['z'][:]
zi = f.variables['zi'][:]
rho = f.variables['rho'][:]  # m
rhow2 = f.variables['rhoi'][:]  # m
p = f.variables['p'][:]  # m

f.close()

rhow = np.zeros(rhow2.shape[0] + 1)
rhow[:-1] = rhow2
rhow[-1] = rhow2[-1]

#The zi in file is for some reason wrong in the sense that it is not always between the z levels). Not sure what to use
zi_from_marat = np.array ([0.0000000, 40., 82.40000, 127.3440, 174.9846,
   225.4837,       279.0128,       335.7535,       395.8987,       459.6526,
   527.2318,       598.8657,       674.7977,       755.2855,       840.6027,
   931.0388,       1026.901,       1129.591,       1242.550,       1366.805,
   1503.486,       1653.834,       1819.218,       2001.140,       2201.311,
   2431.507,       2696.233,       3000.668,       3350.769,       3753.384,
   4216.392,       4716.392,       5216.392,       5716.392,       6216.392,
   6716.392,       7216.392,       7716.392,       8216.392,       8716.392,
   9216.392,       9716.392,       10216.39,       10716.39,      11216.39,
   11716.39,       12216.39,       12716.39,       13216.39,       13716.39,
   14216.39,       14716.39,       15216.39,       15716.39,       16216.39,
   16716.39,       17216.39,       17716.39,       18216.39,       18747.30,
   19354.06,       20047.50,       20840.00,       21745.71,       22780.81,
   23963.79,       25315.76,       26815.76,       28315.76,       29815.76,
   31315.76,       32815.76,       34315.76,       35815.76,       37315.76])

zi = zi_from_marat
print('Marat printed zi that might have an unclear error - therefore, changed manually (data from Marat)')


nzm = z.shape[0]
lon_size = lon.shape[0]
lat_size = lat.shape[0]
nz_size = z.shape[0]


mu = np.zeros(lat_size)
ady = np.zeros(lat_size)
muv = np.zeros(lat_size + 1)

#from setparam
ny_gl = lat.shape[0]
dy = 179.98 / ny_gl
earth_factor = 1
rad_earth = 6371229
deg2rad = np.pi / 180
dy = dy * deg2rad * rad_earth / earth_factor
for j in range(lat_size):
    mu[j] = np.cos(lat[j] * deg2rad)

mu_extend = np.zeros(lat_size + 4)
mu_extend[2:-2] = mu
mu_extend[1] = mu_extend[2] / 2  
mu_extend[0] = mu_extend[1] / 2 
mu_extend[-2] = mu_extend[-3]/2
mu_extend[-1] = mu_extend[-2]/2

latv_gl_high = np.zeros(lat_high.shape[0] + 1)
latv_gl_high[0] = -90
latv_gl_high[-1] = 90

for j in range(1,latv_gl_high.shape[0]-1):
    latv_gl_high[j] = 0.5*(lat_high[j]  + lat_high[j-1])

yv_gl_glob_2_high = latv_gl_high[:]*deg2rad*rad_earth/earth_factor    

dy_high = 179.98 / lat_high.shape[0]
dy_high = dy_high * deg2rad * rad_earth / earth_factor

y_gl_glob_2_high= np.zeros(lat_high.shape[0])
for j in range(lat_high.shape[0]):
    y_gl_glob_2_high[j] = 0.5*(yv_gl_glob_2_high[j + 1]+yv_gl_glob_2_high[j])


##from setgrid
dy2_high = y_gl_glob_2_high[int(lat_high.shape[0]/2)]-y_gl_glob_2_high[int(lat_high.shape[0]/2 - 1)]
ady_glob_high3 = np.zeros(lat_high.shape[0])
for j in range( lat_high.shape[0]):
    ady_glob_high3[j] = (yv_gl_glob_2_high[j+1]-yv_gl_glob_2_high[j])/dy2_high
print('ady_glob_high3 is the correct one to use for calculating the j_start and j_end')

j_start,j_end = cfunc.calc_y_ind_edge_processor(ady_glob_high3, lat_high.shape[0], int(lat_size), ny_coarse_proc=29, coarse_fact=12, processor_points=96)


# Calculating coarse topography related quantities.
path_terra_coare = '/ocean/projects/ees220005p/janniy/python_fortran_coarse_graining/f2py_global_sam/files_coarse_JY/'
terra_path = 'coarse_points.nc4'
if not os.path.isfile(path_terra_coare + terra_path):
    # Create files
    terra_points = np.zeros([lon_size, lat_size, nz_size])
    terra_tot = np.zeros([lon_size, lat_size, nz_size])
    for k in range(nz_size):
        terra_w_path = '/ocean/projects/ees220005p/janniy/gsam_data/high_res_snapshot/DYAMOND2_9216x4608x74_10s_4608_TERR_MASKS_TERRA.atm.3D.nc'
        slice_ter_tot_surf = dict()
        d = Dataset(terra_w_path, 'r')
        data = d.variables['TERRA']
        slice_ter_tot_surf['TERRA'] = data[:, k, :, :]
        del data
        d.close()

        terra_mvax = np.moveaxis(slice_ter_tot_surf['TERRA'], (0, 1, 2), (2, 1, 0))
        aa, bb = terra_num_of_points(terra_mvax, lon_size, lat_size, k, k + 1, j_start, j_end, coarseness=12)
        terra_points[:, :, k] = aa[:, :, 0]
        terra_tot[:, :, k] = bb[:, :, 0]
        print('SAVE TERRA points (TODO)')
        lat_coarse = lat
        lon_coarse = lon

    netxarr = xr.DataArray(
        data=np.moveaxis(terra_points, (0, 1, 2), (2, 1, 0)).astype(np.float32),
        dims=['z', 'lat_coarse', 'lon_coarse'],
        coords=dict(
            z=('z', z),
            lat_coarse=('lat_coarse', lat_coarse),
            lon_coarse=('lon_coarse', lon_coarse),
        ),
        attrs=dict(
            description='Number of point above terra',
            units="number",
        )
    )

    netxarr = netxarr.rename('Terra_points')
    netset = netxarr.to_dataset()

    netxarr2 = xr.DataArray(
        data=np.moveaxis(terra_tot, (0, 1, 2), (2, 1, 0)).astype(np.float32),
        dims=['z', 'lat_coarse', 'lon_coarse'],
        coords=dict(
            z=('z', z),
            lat_coarse=('lat_coarse', lat_coarse),
            lon_coarse=('lon_coarse', lon_coarse),
        ),
        attrs=dict(
            description='Total Number of point in coarse graining',
            units="number",
        )
    )
    netset['Terra_tot'] = netxarr2
    netset.to_netcdf(path=path_terra_coare + terra_path, mode='w', format='NETCDF4')


topog_path_mean = 'topog_cg.nc4'

if not os.path.isfile(path_terra_coare + topog_path_mean):
    mu_gl_high = np.zeros(lat_high.shape[0])

    for j in range(lat_high.shape[0]):
        mu_gl_high[j] = np.cos(lat_high[j] * deg2rad)

    topog_coarse_mean, topog_coarse_median = coarse_grain_topog_median_mean(
        np.moveaxis(topog_high, (0, 1, 2), (2, 1, 0))[:, :, 0], mu_gl_high, ady_glob_2_high, lon_size,
        lat_size, j_start, j_end, coarseness=12)

    topog_mean = topog_coarse_mean
    topog_median = topog_coarse_median

    terra_mean = np.zeros([topog_mean.shape[0], topog_mean.shape[1], z.shape[0]]) + 1
    for i in range(terra_mean.shape[0]):
        for j in range(terra_mean.shape[1]):
            for k in range(z.shape[0]):
                if topog_mean[i, j] > z[k]:
                    terra_mean[i, j, k] = 0
                else:
                    #                 terra[k:,jj,ii] = 1
                    break
    ind_start_terra = 74 - np.sum(terra_mean, axis=2)

    terra_median = np.zeros([topog_mean.shape[0], topog_mean.shape[1], z.shape[0]]) + 1
    for i in range(terra_median.shape[0]):
        for j in range(terra_median.shape[1]):
            for k in range(z.shape[0]):
                if topog_mean[i, j] > z[k]:
                    terra_median[i, j, k] = 0
                else:
                    #                 terra[k:,jj,ii] = 1
                    break

    ind_start_terra_median = 74 - np.sum(terra_median, axis=2)


    path_terra_coare = '/ocean/projects/ees220005p/janniy/python_fortran_coarse_graining/f2py_global_sam/files_coarse_JY/'
    topog_path_mean = 'topog_cg.nc4'
    netxarr = xr.DataArray(
        data=np.moveaxis(topog_coarse_mean, (0, 1), (1, 0)).astype(np.float32),
        dims=['lat_coarse', 'lon_coarse'],
        coords=dict(
            lat_coarse=('lat_coarse', lat_coarse),
            lon_coarse=('lon_coarse', lon_coarse),
        ),
        attrs=dict(
            description='Coarse grained topography (mean)',
            units="m",
        )
    )

    netxarr = netxarr.rename('topog_coarse_mean')
    netset = netxarr.to_dataset()

    netxarr2 = xr.DataArray(
        data=np.moveaxis(topog_coarse_median, (0, 1), (1, 0)).astype(np.float32),
        dims=['lat_coarse', 'lon_coarse'],
        coords=dict(
            lat_coarse=('lat_coarse', lat_coarse),
            lon_coarse=('lon_coarse', lon_coarse),
        ),
        attrs=dict(
            description='Coarse grained topography (meadian)',
            units="m",
        )
    )
    netset['topog_coarse_median'] = netxarr2

    netxarr3 = xr.DataArray(
        data=np.moveaxis(ind_start_terra, (0, 1), (1, 0)).astype(np.float32),
        dims=['lat_coarse', 'lon_coarse'],
        coords=dict(
            lat_coarse=('lat_coarse', lat_coarse),
            lon_coarse=('lon_coarse', lon_coarse),
        ),
        attrs=dict(
            description='First index above surface (mean)',
            units="m",
        )
    )
    netset['ind_start_terra_mean'] = netxarr3

    netxarr3 = xr.DataArray(
        data=np.moveaxis(ind_start_terra_median, (0, 1), (1, 0)).astype(np.float32),
        dims=['lat_coarse', 'lon_coarse'],
        coords=dict(
            lat_coarse=('lat_coarse', lat_coarse),
            lon_coarse=('lon_coarse', lon_coarse),
        ),
        attrs=dict(
            description='First index above surface (median)',
            units="m",
        )
    )
    netset['ind_start_terra_median'] = netxarr3

    netset.to_netcdf(path=path_terra_coare + topog_path_mean, mode='w', format='NETCDF4')




#Read coarse topography related variables
topog_path_tot = path_terra_coare + topog_path_mean
f_toppg = Dataset(topog_path_tot , mode='r')
topog_mean = np.moveaxis(f_toppg.variables['topog_coarse_mean'][:],(0,1),(1,0))  # m
topog_median = np.moveaxis(f_toppg.variables['topog_coarse_median'][:],(0,1),(1,0))  # m
terra_ind_mean = np.moveaxis(f_toppg.variables['ind_start_terra_mean'][:],(0,1),(1,0))  # m
terra_ind_median = np.moveaxis(f_toppg.variables['ind_start_terra_median'][:],(0,1),(1,0))  # m
f_toppg.close()

print('Calculating the new sigma coordinate array for interpolation')

sigma_reference = p / p[0]  
sigma_reference = sigma_reference.data
sigma_tot = np.zeros([lon_size,lat_size,z.shape[0]])
sigma_first_z_index = np.zeros([lon_size,lat_size])
for i in range(lon_size):
    for j in range(lat_size):
        first_z_ind = int(terra_ind_mean[i, j])
        sigma_first_z_index[i, j] = p[first_z_ind]
        if first_z_ind == 0:
            sigma_tot[i, j, :] = sigma_reference
        else:
            sigma_tot[i, j, first_z_ind:] = p[first_z_ind:] / p[first_z_ind]

print('CHECK which ady to use! I should check which one reproduces the high res? OR is it actually related to how I want to define the grid of the low res?')

latv_gl = np.zeros(lat.shape[0] + 1)
latv_gl[0] = -90
latv_gl[-1] = 90

for j in range(1,latv_gl.shape[0]-1):
    latv_gl[j] = 0.5*(lat[j]  + lat[j-1])


#Option 1:
yv_gl_glob = lat[:]*deg2rad*rad_earth/earth_factor    #I use lat possibly need to shift by 0.5 grid
ady_glob = np.zeros(lat.shape[0]-1)
for j in range( lat.shape[0]-1):
    ady_glob[j] = (yv_gl_glob[j+1]-yv_gl_glob[j])/dy


# # #Option 2:
yv_gl_glob_2 = latv_gl[:]*deg2rad*rad_earth/earth_factor    #I use lat possibly need to shift by 0.5 grid
ady_glob_2 = np.zeros(lat.shape[0])


y_gl_glob_2= np.zeros(lat.shape[0])
for j in range(lat.shape[0]):
    y_gl_glob_2[j] = 0.5*(yv_gl_glob_2[j + 1]+yv_gl_glob_2[j])


##from setgrid
dy2 = y_gl_glob_2[int(lat.shape[0]/2)]-y_gl_glob_2[int(lat.shape[0]/2 - 1)]


for j in range( lat.shape[0]):
    ady_glob_2[j] = (yv_gl_glob_2[j+1]-yv_gl_glob_2[j])/dy2
ady = ady_glob_2

print('change ady_glob_2 since it should have a different dy')

mu_glob = np.zeros(lat.shape[0])
for j in range(lat.shape[0]):
    mu_glob[j] = np.cos(lat[j]* deg2rad)
mu = mu_glob

ady_extend = np.zeros(lat_size + 4)
ady_extend[2:-2] = ady
ady_extend[1] = ady_extend[2] + (ady_extend[2] - ady_extend[3])
ady_extend[0] = ady_extend[1] + (ady_extend[1] - ady_extend[2])
ady_extend[-2] = ady_extend[-3]*2 - ady_extend[-4]
ady_extend[-1] = ady_extend[-2]*2 - ady_extend[-3]

# print('change calculation of muv to follow gsam')
for j in range(1,lat_size):
    # Cecck muv_global in setgrid.f90
    muv[j] = (ady[j-1] * mu[j] + ady[j] * mu[j-1]) / (ady[j-1]  +  ady[j])
muv[0] = np.cos(latv_gl[0]* deg2rad)
muv[-1] = np.cos(latv_gl[-1]* deg2rad)

print('I am not sure if I did the correct shift for muv. It depends in how v is printed I guess')
muv_extend = np.zeros(lat_size + 4)
muv_extend[2:-1] = muv
muv_extend[1] = muv_extend[2] / 2  
muv_extend[0] = muv_extend[1] / 2  
muv_extend[-1] = muv_extend[-2]/2


print('To change boundaries when fo terms when I use it later')

nx_gl = lon.shape[0]
dx = 360./nx_gl
latitude0 = 0
dx = dx*deg2rad*rad_earth/earth_factor*np.cos(deg2rad*latitude0)

dz = zi[1] - zi[0]

adzw = np.zeros(nz_size + 1)
adz = np.zeros(nz_size)
for k in range(nz_size):
    adz[k] = (zi[k + 1] - zi[k]) / dz

for k in range(1, nz_size):
    adzw[k] = (z[k] - z[k - 1]) / dz

adzw[0] = 1.
adzw[nz_size] = adzw[nz_size - 1]


cp = 1004.       #      % Specific heat of air, J/kg/K
ggr = 9.81        #     % Gravity acceleration, m/s2
lcond = 2.5104e+06 #    % Latent heat of condensation, J/kg
lfus = 0.3336e+06   #   % Latent heat of fusion, J/kg
lsub = 2.8440e+06 #      % Latent heat of sublimation, J/kg
rv = 461. #              % Gas constant for water vapor, J/kg/K
rgas = 287. #            % Gas constant for dry air, J/kg/K
diffelq = 2.21e-05#     % Diffusivity of water vapor, m2/s
therco = 2.40e-02#      % Thermal conductivity of air, J/m/s/K
muelq = 1.717e-05#      % Dynamic viscosity of air

fac_cond = lcond/cp
fac_fus = lfus/cp
fac_sub = lsub/cp

tbgmin = 253.16#;    % Minimum temperature for cloud water., K
tbgmax = 273.16#;    % Maximum temperature for cloud ice, K
tprmin = 268.16#;    % Minimum temperature for rain, K
tprmax = 283.16#;    % Maximum temperature for snow+graupel, K
tgrmin = 223.16#;    % Minimum temperature for snow, K
tgrmax = 283.16#;    % Maximum temperature for graupel, K

a_pr = 1./(tprmax-tprmin)

# Initialize
file_times = np.arange(start_time, end_time + interval, interval)
n_files = np.size(file_times)

print('yani11')
# Loop over files
for ifile, file_time in enumerate(file_times): # loop over
    #breakpoint()
    filename_wildcard = loaddir  + experiment + '*' + str(file_time).zfill(10)  + '.atm.3DC.nc'  # New version of data
    filename_wildcard_2D = loaddir  + experiment + '*' + str(file_time).zfill(10)  + '.2DC_atm.nc' 

    filename_coarse = glob.glob(filename_wildcard)
    filename_coarse_2D = glob.glob(filename_wildcard_2D)
    print(filename_coarse[0])

    file_name_orig = filename_coarse[0][filename_coarse[0].index(experiment) + 0:]

    print('verfify that the coarse-grained variables were not calculated:')
    f_tmp = Dataset(filename_coarse[0], mode='r')
    if 'rhoQTW_resolved' in f_tmp.variables.keys():
        print('variable exists, skipping iteration')
        continue  # skip analysis if file exists


    #Read LOWRES
    f = Dataset(filename_coarse[0], mode='r')
    # varnammes = ['U', 'V', 'W', 'QV', 'QC', 'QI', 'TABS', 'QG', 'QR', 'QS']
    u = f.variables['U'][:]
    v = f.variables['V'][:]
    w = f.variables['W'][:]
    #I think that Marat did not multiply by 1000 for the coarse fields.
    qg = f.variables['QG'][:]  #/1000
    qr = f.variables['QR'][:] #/ 1000
    qs = f.variables['QS'][:] #/ 1000
    qv = f.variables['QV'][:] #/ 1000
    qc = f.variables['QC'][:] #/ 1000
    qi = f.variables['QI'][:] #/ 1000
    tabs = f.variables['TABS'][:]
    t = f.variables['T'][:]

    f.close()

    qn = qc + qi
    qp = qr + qg + qs
    qt = qn + qv 

    u = cfunc.squeeze_reshape_var(u)
    v = cfunc.squeeze_reshape_var(v)
    w = cfunc.squeeze_reshape_var(w)
    qt = cfunc.squeeze_reshape_var(qt)
    qv = cfunc.squeeze_reshape_var(qv)
    qn = cfunc.squeeze_reshape_var(qn)
    qp = cfunc.squeeze_reshape_var(qp)
    tabs = cfunc.squeeze_reshape_var(tabs)
    t = cfunc.squeeze_reshape_var(t)


    qc = cfunc.squeeze_reshape_var(qc)
    qr = cfunc.squeeze_reshape_var(qr)
    qg = cfunc.squeeze_reshape_var(qg)
    qi = cfunc.squeeze_reshape_var(qi)
    qs = cfunc.squeeze_reshape_var(qs)
    end = time.time()
    print('read 3D coarse')
    print(end - start)

    gamaz = np.zeros(tabs.shape)
    gamaz1d = np.zeros(tabs.shape[2])
    k_ind = -1
    for k in range(nz_size):
        k_ind = k_ind + 1
        gamaz[:, :, k_ind] = ggr / cp * z[k]
        gamaz1d[k_ind] = ggr / cp * z[k]
    
    # change tfull to marat's T
    #tfull = tabs + gamaz - fac_cond * (qc + qr) - fac_sub * (qg + qi + qs)
    tfull = t

    dimx = qp.shape[0]
    dimy= qp.shape[1]
    dimz = qp.shape[2]
    dimzw = dimz +1

    q_flux_x_out = np.zeros((dimx,dimy,dimz))
    q_flux_y_out = np.zeros((dimx,dimy,dimz))
    q_flux_z_out = np.zeros((dimx,dimy,dimz))

    qp_flux_x_out = np.zeros((dimx,dimy,dimz))
    qp_flux_y_out = np.zeros((dimx,dimy,dimz))
    qp_flux_z_out = np.zeros((dimx,dimy,dimz))
    
    qt_flux_x_out = np.zeros((dimx,dimy,dimz))
    qt_flux_y_out = np.zeros((dimx,dimy,dimz))
    qt_flux_z_out = np.zeros((dimx,dimy,dimz))

    t_flux_x_out = np.zeros((dimx,dimy,dimz))
    t_flux_y_out = np.zeros((dimx,dimy,dimz))
    t_flux_z_out = np.zeros((dimx,dimy,dimz))

    tfull_flux_x_out = np.zeros((dimx,dimy,dimz))
    tfull_flux_y_out = np.zeros((dimx,dimy,dimz))
    tfull_flux_z_out = np.zeros((dimx,dimy,dimz))

    q_adv_tend_out= np.zeros((dimx,dimy,dimz))
    qp_adv_tend_out= np.zeros((dimx,dimy,dimz))
    t_adv_tend_out= np.zeros((dimx,dimy,dimz))
    tfull_adv_tend_out= np.zeros((dimx,dimy,dimz))

    # Create fields with extra data at the boundary...
    ext_dim = 4
    u_bound = cfunc.reshape_add_bound_dims(u,ext_dim)
    v_bound = cfunc.reshape_add_bound_dims(v,ext_dim)
    w2 = np.zeros((w.shape[0],w.shape[1],w.shape[2]+1))
    w2[:,:,:-1] = w
    w_bound = cfunc.reshape_add_bound_dims(w2,ext_dim)
    qt_bound = cfunc.reshape_add_bound_dims(qt,ext_dim)
    qp_bound = cfunc.reshape_add_bound_dims(qp,ext_dim)
    tfull_bound = cfunc.reshape_add_bound_dims(tfull,ext_dim)

    # % Step 02:07: Advect variables
    print('check if I need to calc ady differently (I think it should be done differently... )')
    start = time.time()
    advect_scalar3D_f2py_my_min2.advect_scalar3d_f2py(qt_bound, u_bound, v_bound, w_bound, rho,
                                                      rhow, dx, dy, dz, dtn, adz, ady_extend, mu_extend,
                                                      muv_extend, q_flux_x_out,
                                                      q_flux_y_out, q_flux_z_out, q_adv_tend_out, dosubtr=True,
                                                      my_val=True, val_min=0)

    qt = qt  + q_adv_tend_out*dtn
    qt = np.maximum(qt,0.0)
    print('afsasdf')
    advect_scalar3D_f2py_my_min2.advect_scalar3d_f2py(qp_bound, u_bound, v_bound, w_bound, rho,
                                                      rhow, dx, dy, dz, dtn, adz, ady_extend, mu_extend,
                                                      muv_extend, qp_flux_x_out,
                                                      qp_flux_y_out, qp_flux_z_out, qp_adv_tend_out, dosubtr=True,
                                                      my_val=True, val_min=0)
    qp = qp  + qp_adv_tend_out*dtn
    qp = np.maximum(qp,0.0)

    advect_scalar3D_f2py_my_min2.advect_scalar3d_f2py(tfull_bound, u_bound, v_bound, w_bound, rho,
                                                      rhow, dx, dy, dz, dtn, adz, ady_extend, mu_extend,
                                                      muv_extend, tfull_flux_x_out,
                                                      tfull_flux_y_out, tfull_flux_z_out, tfull_adv_tend_out, dosubtr=True,
                                                      my_val=True, val_min=0)

    tfull = tfull  + tfull_adv_tend_out*dtn


    print('May need to approximate the error that I do not directly calculate the advection of H_L instead of h_L')

    omp_3d  = np.maximum(0.,np.minimum(1.,(tabs-tprmin)*a_pr)) #calculate the effective latent heat of precipitation because tabs was changed in cloud?
    fac_dqp_tfull = (fac_cond+(1.-omp_3d)*fac_fus) #This is approximated because I assume that the temperature

    t_flux_z_out   = tfull_flux_z_out  +  fac_dqp_tfull * qp_flux_z_out

    end = time.time()
    print(end - start)

    f = Dataset(filename_coarse[0], mode='r')
    tfull_flux_z_out_coarse = f.variables['rhoTW'][:]
    # I think Marat did not multiply by 1000
    print('Currently the qp,q fluxes are very different than what I calculate in the federov case... ')
    qp_flux_z_out_coarse = f.variables['rhoQPW'][:] #/ 1000
    q_flux_z_out_coarse = f.variables['rhoQTW'][:] #/ 1000

    tfull_flux_z_out_coarse = cfunc.squeeze_reshape_var(tfull_flux_z_out_coarse)
    qp_flux_z_out_coarse = cfunc.squeeze_reshape_var(qp_flux_z_out_coarse)
    q_flux_z_out_coarse = cfunc.squeeze_reshape_var(q_flux_z_out_coarse)

    # There is an inacuracy due to the usage of
    t_flux_z_out_coarse = tfull_flux_z_out_coarse  + qp_flux_z_out_coarse*fac_dqp_tfull

    f.close()


    # % Step 04:01:  Calculate subgrid terms
    # advection:
    t_flux_z_out_subgrid = cfunc.calc_subgrid(t_flux_z_out_coarse, t_flux_z_out)
    tfull_flux_z_out_subgrid = cfunc.calc_subgrid(tfull_flux_z_out_coarse, tfull_flux_z_out)
    q_flux_z_out_subgrid = cfunc.calc_subgrid(q_flux_z_out_coarse, q_flux_z_out)
    qp_flux_z_out_subgrid = cfunc.calc_subgrid(qp_flux_z_out_coarse, qp_flux_z_out)

    print('I want to interpolate to the reference sigma coordinate here. Once with mean topography once with median topography')
    print('Fields to interpolate: All outputs and inputs!,topog_mean,topog_median')


    tabs_sigma = np.zeros(tabs.shape)
    qt_sigma = np.zeros(qt.shape)
    qp_sigma = np.zeros(qp.shape)
    tfull_flux_z_out_subgrid_sigma = np.zeros(tfull.shape)
    t_flux_z_out_subgrid_sigma = np.zeros(tfull.shape)
    q_flux_z_out_subgrid_sigma = np.zeros(qt.shape)
    qp_flux_z_out_subgrid_sigma = np.zeros(qp.shape)
    rho_sigma = np.zeros(qt.shape)

    print('reading coarse grained to interpolate')
    f = Dataset(filename_coarse[0], mode='r')
    KEDDYSC = f.variables['KEDDYSC'][:]
    QRAD = f.variables['QRAD'][:]
    PREC = f.variables['PREC'][:]
    LPREC = f.variables['LPREC'][:]
    SED = f.variables['SED'][:]
    LSED = f.variables['LSED'][:]
    QP_MICRO = f.variables['QP_MICRO'][:]
    RHOQPW = f.variables['rhoQPW'][:]
    RHOQPS = f.variables['rhoQPWS'][:]
    T = f.variables['T'][:]
    QV = f.variables['QV'][:]
    QC = f.variables['QC'][:]
    QI = f.variables['QI'][:]

    f.close()
    
    f = Dataset(filename_coarse_2D[0], mode='r')
    
    SKT = f.variables['SKT'][:]
    
    f.close()
    
    KEDDYSC = cfunc.squeeze_reshape_var(KEDDYSC)
    QRAD = cfunc.squeeze_reshape_var(QRAD)
    PREC = cfunc.squeeze_reshape_var(PREC)
    LPREC = cfunc.squeeze_reshape_var(LPREC)
    SED = cfunc.squeeze_reshape_var(SED)
    LSED = cfunc.squeeze_reshape_var(LSED)
    QP_MICRO = cfunc.squeeze_reshape_var(QP_MICRO)
    RHOQPW = cfunc.squeeze_reshape_var(RHOQPW)
    RHOQPS = cfunc.squeeze_reshape_var(RHOQPS)
    T = cfunc.squeeze_reshape_var(T)
    QV = cfunc.squeeze_reshape_var(QV)
    QC = cfunc.squeeze_reshape_var(QC)
    QI = cfunc.squeeze_reshape_var(QI)
    SKT = np.swapaxes(np.squeeze(SKT),0,1)



    keddysc_sigma = np.zeros(KEDDYSC.shape)
    qrad_sigma = np.zeros(QRAD.shape)
    prec_sigma = np.zeros(PREC.shape)
    lprec_sigma = np.zeros(LPREC.shape)
    sed_sigma = np.zeros(SED.shape)
    lsed_sigma = np.zeros(LSED.shape)
    qp_micro_sigma = np.zeros(QP_MICRO.shape)
    rhoqpw_sigma = np.zeros(RHOQPW.shape)
    rhoqps_sigma = np.zeros(RHOQPS.shape)
    T_sigma = np.zeros(T.shape)
    QV_sigma = np.zeros(QV.shape)
    QC_sigma = np.zeros(QC.shape)
    QI_sigma = np.zeros(QI.shape)
    SKT_sigma = np.zeros(SKT.shape)

    print('interpolation')
    start1 = time.time()
    for i in range(lon_size):
        for j in range(lat_size):
            first_z_ind = int(terra_ind_mean[i,j])
            dum_i = 0
            while tabs[i,j,first_z_ind] == 0: #this is a fix to prevent the lower level to be where the temperature is not defined
                first_z_ind = first_z_ind  + 1
                dum_i = dum_i + 1
                if dum_i > 2:
                    print(dum_i,i,j)
            if first_z_ind == 0:
                tabs_sigma[i, j, :] = tabs[i, j, :]
                qt_sigma[i, j, :] = qt[i, j, :]
                qp_sigma[i, j, :] = qp[i, j, :]
                tfull_flux_z_out_subgrid_sigma[i, j, :] = tfull_flux_z_out_subgrid[i, j, :]
                t_flux_z_out_subgrid_sigma[i, j, :] = t_flux_z_out_subgrid[i, j, :]
                q_flux_z_out_subgrid_sigma[i, j, :] = q_flux_z_out_subgrid[i, j, :]
                qp_flux_z_out_subgrid_sigma[i, j, :] = qp_flux_z_out_subgrid[i, j, :]
                rho_sigma[i, j, :] = rho
                keddysc_sigma[i, j, :] = KEDDYSC[i, j, :]
                qrad_sigma[i, j, :] = QRAD[i, j, :]
                prec_sigma[i, j, :] = PREC[i, j, :]
                lprec_sigma[i, j, :] = LPREC[i, j, :]
                sed_sigma[i, j, :] = SED[i, j, :]
                lsed_sigma[i, j, :] = LSED[i, j, :]
                qp_micro_sigma[i, j, :] = QP_MICRO[i, j, :]
                rhoqpw_sigma[i, j, :] = RHOQPW[i, j, :]
                rhoqps_sigma[i, j, :] = RHOQPS[i, j, :]
                T_sigma[i, j, :] = T[i, j, :]
                QV_sigma[i, j, :] = QV[i, j, :]
                QC_sigma[i, j, :] = QC[i, j, :]
                QI_sigma[i, j, :] = QI[i, j, :]
            else:
                tabs_sigma[i, j, :] = np.interp(np.flip(sigma_reference),np.flip(sigma_tot[i, j, first_z_ind:]),np.flip(tabs[i,j,first_z_ind:]))
                tabs_sigma[i, j, :] = np.flip(tabs_sigma[i, j, :])


                qt_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(qt[i, j, first_z_ind:]))
                qt_sigma[i, j, :] = np.flip(qt_sigma[i, j, :])
                
                qp_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(qp[i, j, first_z_ind:]))
                qp_sigma[i, j, :] = np.flip(qp_sigma[i, j, :])
                
                tfull_flux_z_out_subgrid_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(tfull_flux_z_out_subgrid[i, j, first_z_ind:]))
                tfull_flux_z_out_subgrid_sigma[i, j, :] = np.flip(tfull_flux_z_out_subgrid_sigma[i, j, :])
                
                t_flux_z_out_subgrid_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(t_flux_z_out_subgrid[i, j, first_z_ind:]))
                t_flux_z_out_subgrid_sigma[i, j, :] = np.flip(t_flux_z_out_subgrid_sigma[i, j, :])
                
                q_flux_z_out_subgrid_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(q_flux_z_out_subgrid[i, j, first_z_ind:]))
                q_flux_z_out_subgrid_sigma[i, j, :] = np.flip(q_flux_z_out_subgrid_sigma[i, j, :])
                qp_flux_z_out_subgrid_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(qp_flux_z_out_subgrid[i, j, first_z_ind:]))
                qp_flux_z_out_subgrid_sigma[i, j, :] = np.flip(qp_flux_z_out_subgrid_sigma[i, j, :])
                #qt_flux_z_out_subgrid_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                #np.flip(qt_flux_z_out_subgrid[i, j, first_z_ind:]))
                #qt_flux_z_out_subgrid_sigma[i, j, :] = np.flip(qt_flux_z_out_subgrid_sigma[i, j, :])
                rho_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(rho[first_z_ind:]))
                rho_sigma[i, j, :] = np.flip(rho_sigma[i, j, :])
                keddysc_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(KEDDYSC[i, j, first_z_ind:]))
                                              
                keddysc_sigma[i, j, :] = np.flip(keddysc_sigma[i, j, :])
                
                qrad_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(QRAD[i, j, first_z_ind:]))
                qrad_sigma[i, j, :] = np.flip(qrad_sigma[i, j, :])
                prec_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(PREC[i, j, first_z_ind:]))
                prec_sigma[i, j, :] = np.flip(prec_sigma[i, j, :])
                lprec_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(LPREC[i, j, first_z_ind:]))
                lprec_sigma[i, j, :] = np.flip(lprec_sigma[i, j, :])
                
                sed_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(SED[i, j, first_z_ind:]))
                sed_sigma[i, j, :] = np.flip(sed_sigma[i, j, :])
                lsed_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(LSED[i, j, first_z_ind:]))
                lsed_sigma[i, j, :] = np.flip(lsed_sigma[i, j, :])
                
                qp_micro_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(QP_MICRO[i, j, first_z_ind:]))
                qp_micro_sigma[i, j, :] = np.flip(qp_micro_sigma[i, j, :])
                rhoqpw_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(RHOQPW[i, j, first_z_ind:]))
                rhoqpw_sigma[i, j, :] = np.flip(rhoqpw_sigma[i, j, :])
                
                rhoqps_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(RHOQPS[i, j, first_z_ind:]))
                rhoqps_sigma[i, j, :] = np.flip(rhoqps_sigma[i, j, :])   
                T_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(T[i, j, first_z_ind:]))
                T_sigma[i, j, :] = np.flip(T_sigma[i, j, :])
                QC_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(QC[i, j, first_z_ind:]))
                QC_sigma[i, j, :] = np.flip(QC_sigma[i, j, :])
                QV_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(QV[i, j, first_z_ind:]))
                QV_sigma[i, j, :] = np.flip(QV_sigma[i, j, :])
                QI_sigma[i, j, :] = np.interp(np.flip(sigma_reference), np.flip(sigma_tot[i, j, first_z_ind:]),
                                                np.flip(QI[i, j, first_z_ind:]))
                QI_sigma[i, j, :] = np.flip(QI_sigma[i, j, :])

    print('save')
    end1 = time.time()
    print('time of interpolation')
    print(end1-start1)
    netset = xr.open_dataset(filename_coarse[0])

    var_dict = cfunc.create_var_dict()
    unit1 = '_units'
    desc1 = '_desc'
    var_name = 'tfull_flux_z_out_subgrid'
    var = tfull_flux_z_out_subgrid
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)

    var_name = 't_flux_z_out_subgrid'
    var = t_flux_z_out_subgrid
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)

    var_name = 'q_flux_z_out_subgrid'
    var = q_flux_z_out_subgrid
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)


    var_name = 'qp_flux_z_out_subgrid'
    var = qp_flux_z_out_subgrid
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)

    print('need to include a different vertical coordinate')
    var_name = 'tabs_sigma'
    var = tabs_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)

    var_name = 'qt_sigma'
    var = qt_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)


    var_name = 'sigma_tot'
    var = sigma_tot
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qp_sigma'
    var = qp_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'tfull_flux_z_out_subgrid_sigma'
    var = tfull_flux_z_out_subgrid_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 't_flux_z_out_subgrid_sigma'
    var = t_flux_z_out_subgrid_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'q_flux_z_out_subgrid_sigma'
    var = q_flux_z_out_subgrid_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qp_flux_z_out_subgrid_sigma'
    var = qp_flux_z_out_subgrid_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'rho_sigma'
    var = rho_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'keddysc_sigma'
    var = keddysc_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qrad_sigma'
    var = qrad_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'prec_sigma'
    var = prec_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'lprec_sigma'
    var = lprec_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'sed_sigma'
    var = sed_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'lsed_sigma'
    var = lsed_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qp_micro_sigma'
    var = qp_micro_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'rhoqpw_sigma'
    var = rhoqpw_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    
    var_name = 'rhoqps_sigma'
    var = rhoqps_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    
    var_name = 't_sigma'
    var = T_sigma
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qc_sigma'
    var = QC_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qv_sigma'
    var = QV_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'qi_sigma'
    var = QI_sigma
    moisture_flag = True
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'skt'
    var = SKT
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    var_name = 'sfc_reference_p'
    var = sigma_first_z_index
    moisture_flag = False
    cfunc.create_data_array_complete_gsam(var, var_dict[var_name],
                                     var_dict[var_name + desc1], var_dict[var_name + unit1],
                                     lon, lat, z, netset,
                                     filename=filename, moisture_flag=moisture_flag, test_mode=test_mode)
    
    print('saving:')
    netset.to_netcdf(path=savedir + file_name_orig[:-4] + '_resolved' +  '.nc4', mode='w', format='NETCDF4')


