import os
import sys


type_slice="axial"


"""
filename_result="list_all_patients_flair_"+type_slice+".txt"

folder_slices_brats2017_flair_hgg="/data/asclepios/user/pmlynars/slices_bin_normalized_without_registration/T2_FLAIR/BRATS2017/train/HGG/"+type_slice
folder_slices_brats2017_flair_lgg="/data/asclepios/user/pmlynars/slices_bin_normalized_without_registration/T2_FLAIR/BRATS2017/train/LGG/"+type_slice


list_folders=[folder_slices_brats2017_flair_hgg,folder_slices_brats2017_flair_lgg]
"""


folder_slices_brats2017_flair_validation="/data/asclepios/user/pmlynars/slices_bin_normalized_without_registration/T2_FLAIR/BRATS2017/validation/"+type_slice
list_folders=[folder_slices_brats2017_flair_validation]
filename_result="list_all_patients_validation_flair_"+type_slice+".txt"


file_result=open(filename_result,"w")

list_paths_all_patients=[]
for path_folder in list_folders:
	list_patients_one_folder=os.listdir(path_folder)
	list_paths_patients_one_folder=[(path_folder+"/"+patient) for patient in list_patients_one_folder]
	list_paths_all_patients=list_paths_all_patients+list_paths_patients_one_folder

file_result.write("\n".join(list_paths_all_patients))
file_result.close()