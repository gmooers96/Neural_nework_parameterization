import sys
sys.path.append('../../')

import src.training_test_generator_simple_big_data_for_memory_by_parts_numpy_v2 as training_test_generator_simple
import pdb

# path to the coarse-grained data -- note this syntax will select ALL files initially (roughly 1000 files)
filepath = "/ocean/projects/ees220005p/gmooers/GM_Data/**0000[0123]**.nc4"
# where to save the preprocessed data
savepath = "/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/"
# how many vertical levels do you want in the training/test data
levels = 49
# how much of the terrain vector do you want
ground_levels = 1
flag_dict = dict()
# ID for the training data
my_name = "Mini_P1_G_Train_"
rewight_outputs = False
# simulation hour to start at
file_start = 0
# simulation hour to end at
file_end = 160
###

###
# Original Janni Flags I still use
flag_dict['Tin_feature'] = True
flag_dict['qin_feature'] = True
flag_dict['predict_tendencies'] = True
###

###
# Griffin Flag Additions
flag_dict['skt'] = False
flag_dict['land_frac'] = True
flag_dict['sfc_pres'] = True
flag_dict['cos_lat'] = False
flag_dict['sin_lon'] = False
###



training_test_generator_simple.build_training_dataset(filepath=filepath, 
                           savepath=savepath,
                           my_label=my_name,
                           filestart=file_start,
                           fileend=file_end,
                           n_z_input=levels,
                           ground_levels=ground_levels,                           
                           flag_dict=flag_dict, 
                           rewight_outputs = rewight_outputs)

