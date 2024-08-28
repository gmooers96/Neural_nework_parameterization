import sys
sys.path.append('../../')

import src.training_test_generator_simple_big_data_for_memory_by_parts_numpy_v2 as training_test_generator_simple
import pdb

###
filepath = "/ocean/projects/ees220005p/gmooers/GM_Data/**0000[0123]**.nc4"
savepath = "/ocean/projects/ees220005p/gmooers/GM_Data/training_data/Training_Parts/"
levels = 49
ground_levels = 1
flag_dict = dict()
my_name = "Mini_P3_G_Train_"
rewight_outputs = False
#file_start = 320
#file_end = 480
file_start = 8
file_end = 12
###

###
# Original Flags I still use
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

