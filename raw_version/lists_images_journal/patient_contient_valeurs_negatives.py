import sys
import os
import numpy as np

list_modalities=["T2_FLAIR","T1","T1c", "T2"]
#list_patients_flair=open("lists_images/list_flair_patients_training_segmentation.txt","r").read().split("\n")

#list_patients_flair=open("lists_images/miccai_training_patients_flair_axial_with_gt.txt","r").read().split("\n")
#list_patients_flair=open("lists_images/miccai_test_patients_flair_axial_with_gt.txt","r").read().split("\n")
#list_patients_flair=open("lists_images/miccai_training_patients_flair_axial_without_gt.txt","r").read().split("\n")
list_patients_flair=open(sys.argv[1],"r").read().split("\n")

found_error=False

for folder_patient_flair in list_patients_flair:
	#print("je check "+folder_patient_flair)
	for mod in list_modalities:
		filename_slice_i_one_modality=(folder_patient_flair+"/slice50.npy").replace("T2_FLAIR",mod)
		#print("checking "+filename_slice_i_one_modality)
		a=np.load(filename_slice_i_one_modality)
		nb_negative_values=np.sum(np.sum(a<0))

		if(nb_negative_values>0):
		
			print("Negative value in "+filename_slice_i_one_modality)
			found_error=True
			continue
			


if(not(found_error)):
	print("Test ok")