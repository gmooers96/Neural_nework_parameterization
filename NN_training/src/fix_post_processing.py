import post_processing_figures as post_processing_figures
import numpy as np
import pdb

my_z = 49
output_vert_vars = ['Tout', 'T_adv_out','q_adv_out','q_auto_out','q_sed_flux_tot']
save_path =  '/ocean/projects/ees220005p/gmooers/Investigations/Model_Performance/'
nametag = 'EXPERIMENT_0_my_trial_machine_type_cpu_use_poles_True_physical_weighting_False_epochs_1_tr_data_percent_3_num_hidden_layers_1'
single_file = '/ocean/projects/ees220005p/gmooers/GM_Data/DYAMOND2_coars_9216x4608x74_10s_4608_20200120030000_0000001080.atm.3D_resolved.nc4'

truths = np.load("/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Temp_Data/Truth.npy")
preds = np.load("/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Temp_Data/Pred.npy")

post_processing_figures.main_plotting(truth=truths,
                                              pred=preds, 
                                              raw_data=single_file,
                                              z_dim=my_z,
                                             var_names=output_vert_vars,
                                              save_path=save_path,
                                              nametag=nametag,
                                             )