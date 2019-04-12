import theano
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os
import random
import string
from common_functions_theano import *
from neural_network_2D import *

















name_net=sys.argv[1]

path_parameters=sys.argv[2]

name_test=sys.argv[3]

folder_one_fold=sys.argv[4]



nb_classes=int(sys.argv[5])





str_batch_norm=sys.argv[6]
type_slice=sys.argv[7]


str_type_network=sys.argv[8]



perform_postprocessing=False

#perform_postprocessing=True


take_all_slices=True
#take_all_slices=False




pad_zeros_input=True
#target_shape_input_image=[240,240]

#target_shape_input_image=[320,320]
target_shape_input_image=[350,350]





print("VARIABLE NUMBER OF SLICES")




folder_results_all_tests=sys.argv[9]




folder_all_networks="networks/joint2018/"


print("\n\n\n\nUsing "+type_slice+" slices")


folder_all_dices="Dices_cross_validation_August2018/"
os.system("mkdir -p "+folder_all_dices)



if((str_batch_norm!="batch_norm") and (str_batch_norm!='no_batch_norm')):
	print("Argument 7: precise 'batch_norm' or 'no_batch_norm'")
	sys.exit(1)







use_batch_norm=(str_batch_norm=="batch_norm")

if(not(use_batch_norm)):
	print("\n\n\nNo batch normalization\n\n\n")
	print("not implemented")
	sys.exit(0)



if(not(use_batch_norm)):
	multiplication_input=0.01
else:
	multiplication_input=1.0

pad_zeros_input=True
#pad_zeros_input=False


#generate_nii_gz=False
generate_nii_gz=True

save_result=False
#save_result=True






if("test100" in folder_one_fold):
	name_file_paths_folders_flair_patients_test=folder_one_fold+"/list_patients_validation_flair_"+type_slice+".txt"
else:
	name_file_paths_folders_flair_patients_test=folder_one_fold+"/test_patients_flair_"+type_slice+"_with_gt.txt"
	#test_patients_flair_axial_with_gt


print("FILE TEST PATIENTS: "+name_file_paths_folders_flair_patients_test)


















test_enhancing_core=False


if("enhancing_core" in name_test):
	test_enhancing_core=True








#batch_size=5
batch_size=1

print("\n\n\nBATCH SIZE: "+str(batch_size))
print("\n\n\n")


#list_modalities=["T2_FLAIR","T1"]
list_modalities=["T2_FLAIR","T1", "T1c","T2"]






use_global_max_pooling=True
#use_global_max_pooling=False


use_bn_in_fc_layers=False













slice_start_axial=0
slice_end_axial=154

slice_start_sagittal=0
slice_end_sagittal=239

slice_start_coronal=0
slice_end_coronal=239







slice_start_axial_restricted=46
slice_end_axial_restricted=125

slice_start_sagittal_restricted=81
slice_end_sagittal_restricted=160


slice_start_coronal_restricted=65
slice_end_coronal_restricted=189








if(type_slice=="axial"):
	slice_start=slice_start_axial
	slice_end=slice_end_axial

	slice_start_restricted=slice_start_axial_restricted
	slice_end_restricted=slice_end_axial_restricted

elif(type_slice=="sagittal"):
	slice_start=slice_start_sagittal
	slice_end=slice_end_sagittal
	slice_start_restricted=slice_start_sagittal_restricted
	slice_end_restricted=slice_end_sagittal_restricted

elif(type_slice=="coronal"):
	slice_start=slice_start_coronal
	slice_end=slice_end_coronal
	slice_start_restricted=slice_start_coronal_restricted
	slice_end_restricted=slice_end_coronal_restricted
else:
	print("unknown type of slice")
	sys.exit(2)







#constructs lists of paths


file_paths_folders_flair_patients_test=open(name_file_paths_folders_flair_patients_test,"r")
paths_folders_flair_patients_test=file_paths_folders_flair_patients_test.read().split("\n")
file_paths_folders_flair_patients_test.close()


first_image=np.load(paths_folders_flair_patients_test[0]+"/slice0.npy")#reminder: 3 lists: images labeled 1 with gt, images labeled 0, images labeled 1 without gt

#reminder: indexes: patient, list_positive_slices (index 0) or negative slices (index 1), num_slice, list modalitites (index 0) or path_gt, num_modality 
#here we read the first modality


batch_shape=[batch_size,len(list_modalities),first_image.shape[0],first_image.shape[1]]




first_modality=list_modalities[0]



nb_images_gt= batch_size





folder_parameters_for_mean_and_std=path_parameters
apply_bn_conv_layers=use_batch_norm

#create the net



net=NeuralNet2D(folder_all_networks+"/"+name_net+".prototxt", test_phase=True,input_tensor_shape=batch_shape, nb_classes=nb_classes,nb_images_gt=nb_images_gt, str_name=name_net,folder_parameters_for_mean_and_std=folder_parameters_for_mean_and_std,pad_zeros_input=pad_zeros_input,target_shape_input_image=target_shape_input_image,apply_bn_conv_layers=apply_bn_conv_layers)



net.print_layers()



print("definition des fonctions")





print("\n\n\n\nfonctions definies")



print("\n\n\n\n\n\nje lis  les parametres \n\n\n\n\n")
net.set_parameters_all_layers(path_parameters)
print("\n\n\n\n\n\nj'ai bien lu les parametres \n\n\n\n\n")






compute_all_scores=theano.function([net.input_network_0],net.output_segmentation_scores)


num_parameters=int(path_parameters.replace("/","").split("parameters")[-1])

if(pad_zeros_input):
	print("\n\n\nPADDING WITH ZEROS\n\n\n")
	folder_results=folder_results_all_tests+"/"+name_net+"/"+name_test+"_padded_parameters"+str(num_parameters)
else:
	folder_results=folder_results_all_tests+ "/"+name_net+"/"+name_test+"_parameters"+str(num_parameters)



if(perform_postprocessing):
	folder_results=folder_results+"_postprocessed"


if(not(take_all_slices)):
	folder_results=folder_results+"_inner_slices"



folder_results=folder_results+"/"



folder_results_images3D=folder_results+"/images3D"

os.system("mkdir -p "+folder_results)
os.system("mkdir -p "+folder_results_images3D)
patients_display=[]





filename_finish=folder_results+"/finished"


if(os.path.isfile(filename_finish)):
	print("already done")
	sys.exit(0)








def construct_GT_patient(path_patient_labels_axial):

	#initializations
	#nb_total_axial_slices=len(os.listdir(path_patient_labels_axial))
	
	#nb_extract_slices=slice_end- slice_start+1

	first_image=np.load(path_patient_labels_axial+"/slice0.npy")
	height_axial_slice=first_image.shape[0]
	width_axial_slice=first_image.shape[1]

	nb_total_axial_slices=len(os.listdir(path_patient_labels_axial))
	
	#shape_result=[nb_axial_slices, height_axial_slice, width_axial_slice]
	#shape_result=[nb_extract_slices, height_axial_slice, width_axial_slice]
	shape_result=[nb_total_axial_slices, height_axial_slice, width_axial_slice]
		

	#coordinates: (z,y,x)

	result_tensor_3D=np.zeros(shape_result,dtype=np.int32)
	

	#read the labels

	j=-1
	#for num_slice in range(nb_axial_slices):
	for num_slice in range(nb_total_axial_slices):
		j=j+1
		path_image_labels_axial_slice=path_patient_labels_axial+"/slice"+str(num_slice)+".npy"
		
		result_tensor_3D[j,:,:]=np.load(path_image_labels_axial_slice)

		


	return result_tensor_3D



def create_batch_few_slices_one_patient(num_slice_start_batch, batch_size, folder_slices_first_modality_patient, list_modalities):
	first_image_one_channel=np.load(folder_slices_first_modality_patient+"/slice0.npy")

	nb_modalities=len(list_modalities)
	height=first_image_one_channel.shape[0]
	width=first_image_one_channel.shape[1]
	shape_batch=[batch_size, nb_modalities, height, width]

	#initialize tensor
	tensor_images=np.zeros(shape_batch,dtype=theano.config.floatX)

	num_image=-1
	for num_slice in range(num_slice_start_batch,num_slice_start_batch+batch_size):
		num_image=num_image+1
		for num_mod in range(len(list_modalities)):
			mod=list_modalities[num_mod]
			path_img_one_modality=folder_slices_first_modality_patient.replace(first_modality,mod)+"/slice"+str(num_slice)+".npy"
			tensor_images[num_image, num_mod, :, :]=np.load(path_img_one_modality)


	if(multiplication_input!=1.0):
		tensor_images=tensor_images*multiplication_input




	return tensor_images
		







def compute_stats_subclasses_brats_output_one_patient(output_3D_labels, labels_3D_patient):
	#inputs: 3D tensors of integers: dimensions are (z,y,x)  (one 3D volume) 
	
	examples_labeled_as_positive_core=(output_3D_labels==1)+(output_3D_labels==3)
	#not core: edema or healthy
	examples_labeled_as_negative_core=(output_3D_labels==0)+(output_3D_labels==2)
	

	positive_examples_core=(labels_3D_patient==1)+(labels_3D_patient==3)
	negative_examples_core=(labels_3D_patient==0)+(labels_3D_patient==2)

	(precision_core, recall_core, dice_core)=compute_stats_two_masks(examples_labeled_as_positive_core,examples_labeled_as_negative_core, positive_examples_core, negative_examples_core)


	examples_labeled_as_positive_enhancing=(output_3D_labels==3)
	#not core: edema or healthy
	examples_labeled_as_negative_enhancing=(output_3D_labels<3)
	

	positive_examples_enhancing=(labels_3D_patient==3)
	negative_examples_enhancing=(labels_3D_patient<3)

	(precision_enhancing, recall_enhancing, dice_enhancing)=compute_stats_two_masks(examples_labeled_as_positive_enhancing,examples_labeled_as_negative_enhancing, positive_examples_enhancing, negative_examples_enhancing)


	return (precision_core, recall_core, dice_core, precision_enhancing, recall_enhancing, dice_enhancing)





def compute_stats_wt_brats_output_one_patient(output_3D_labels, labels_3D_patient):
	#inputs: 3D tensors of integers: dimensions are (z,y,x)  (one 3D volume) 
	
	examples_labeled_as_positive=(output_3D_labels>0)
	examples_labeled_as_negative=(output_3D_labels==0)
	positive_examples=(labels_3D_patient>0)
	negative_examples=(labels_3D_patient==0)



	precision, recall, dice=compute_stats_two_masks(examples_labeled_as_positive,examples_labeled_as_negative, positive_examples, negative_examples)

	
	return (precision, recall, dice)



def compute_stats_two_masks(examples_labeled_as_positive,examples_labeled_as_negative, positive_examples, negative_examples):
	#each input= 3D integer tensor with ones and zeros
	true_positives=examples_labeled_as_positive*positive_examples
	false_positives=examples_labeled_as_positive*negative_examples
	false_negatives=examples_labeled_as_negative*positive_examples

	nb_true_positives=np.sum(np.sum(np.sum(true_positives)))
	nb_false_positives=np.sum(np.sum(np.sum(false_positives)))
	nb_false_negatives=np.sum(np.sum(np.sum(false_negatives)))


	nb_positive_examples=np.sum(np.sum(np.sum(positive_examples)))
	nb_examples_labeled_as_positive=nb_true_positives+nb_false_positives


	if((nb_true_positives+nb_false_negatives)>0):
		#if(nb_positive_examples>0):
		recall=float(nb_true_positives)/float(nb_true_positives+nb_false_negatives)
	else:
		recall=-1.0

	if((nb_true_positives+nb_false_positives)>0):
		#if(nb_positive_examples>0):
		precision=float(nb_true_positives)/float(nb_true_positives+nb_false_positives)
	else:
		precision=-1.0

	if(nb_positive_examples>0):
		dice=float(2*nb_true_positives)/float(nb_positive_examples+nb_examples_labeled_as_positive)
	else:
		dice=-1.0
	return (precision, recall, dice)



def compute_median_from_list(list_floats):

	if(len(list_floats)==0):
		print("empty list")
		return -1.0

	list_floats.sort()
	if(len(list_floats)%2==0):
		result=(list_floats[(len(list_floats)/2)-1]+list_floats[len(list_floats)/2])/2.0
	else:
		result=list_floats[len(list_floats)/2]


	return result




def compute_output_network(net, folder_slices_first_modality_patient,list_modalities, type_slice):
	#the function which calls this function has to choose a good type of slices (axial/sagittal/coronal)



	#define and compute the shapes
	
	##nb_slices=len(os.listdir(folder_slices_first_modality_patient))
	

	#nb_extract_slices=slice_end- slice_start+1

	nb_slices=len(os.listdir(folder_slices_first_modality_patient))



	nb_extract_slices=slice_end- slice_start+1

	print("slice start: "+str(slice_start))
	print("slice end: "+str(slice_end))
	print("nb_extract_slices=: "+str(nb_extract_slices))









	first_image=np.load(folder_slices_first_modality_patient+"/slice0.npy")

	
	shape_result_before_dimshuffle=[nb_extract_slices,first_image.shape[0], first_image.shape[1]]


	result_before_dimshuffle=np.zeros(shape_result_before_dimshuffle,dtype=np.int32)

	
	

	

	nb_zeros_top_nn=net.nb_zeros_top
	nb_zeros_left_nn=net.nb_zeros_left
	offset_y_nn=net.total_offset_y
	offset_x_nn=net.total_offset_x


	#compute_classification_scores=theano.function([net.input_network],net.output_classification_scores)
	target_shape_outputs_one_batch=[batch_size,first_image.shape[0],first_image.shape[1]]

	








	#for num_slice_start_batch in range(0,nb_extract_slices,batch_size):
	j=-batch_size
	for num_slice_start_batch in range(slice_start,slice_end,batch_size):
		j=j+batch_size
		batch=create_batch_few_slices_one_patient(num_slice_start_batch, batch_size, folder_slices_first_modality_patient, list_modalities)
	




		sc=compute_all_scores(batch)
	

		output_tensor_labels_batch=np.argmax(sc,axis=1)
			
			#output_tensor_labels_batch=add_zero_padding(output_tensor_labels_batch,target_shape_outputs_one_batch)
		output_tensor_labels_batch=adjust_shape_tensor3(output_tensor_labels_batch,target_shape_outputs_one_batch,nb_zeros_top_nn, nb_zeros_left_nn, offset_y_nn,offset_x_nn)
		result_before_dimshuffle[j:(j+batch_size),:,:]=output_tensor_labels_batch

		


	result=result_before_dimshuffle
	return (result)














filename_dice=folder_results+"/dice.txt"
file_dice=open(filename_dice,"w")



mean_dice_wt=0.0
mean_dice_core=0.0
mean_dice_enhancing=0.0

nb_treated_patients_wt=0
nb_treated_patients_core=0
nb_treated_patients_enhancing=0



dices_wt=[]
dices_core=[]
dices_enhancing=[]




dices_wt=[]
dices_core=[]
dices_enhancing=[]



for path_folder_flair_patient in paths_folders_flair_patients_test:


	

	name_patient=path_folder_flair_patient.split("/")[-1]
	if(name_patient==""):
		name_patient=path_folder_flair_patient.split("/")[-2]



	path_patient_first_modality=path_folder_flair_patient.replace("axial",type_slice)


	path_file_nii_gz=folder_results_images3D+"/"+name_patient+".nii.gz"



	result_exists=os.path.isfile(path_file_nii_gz)









	path_gt_axial=path_patient_first_modality.replace(first_modality,"GT").replace(type_slice,"axial")
	
	#path_gt_axial=path_patient_first_modality.replace(first_modality+"/lungs_NIH/","GT/lungs_NIH/gtv/").replace(type_slice,"axial")







	gt_exists=os.path.isdir(path_gt_axial)



	print("net: "+name_net)
	print("Patient: "+path_patient_first_modality)
	print(path_gt_axial)
	
	if(not(result_exists)):
		#create the result nii.gz file
	

		#compute the output of the network on this patient

		#construct the tensor 

		

		#compute the output labels of this network (3D tensor: z, y , x)
		output_3D_current_network=compute_output_network(net, path_patient_first_modality, list_modalities, type_slice)

		


		if(test_enhancing_core):
			#replace labe1 1--> label 3
			output_3D_current_network=output_3D_current_network*3


			
		#reshape
		if(type_slice=="coronal"):
			output_3D_current_network=output_3D_current_network.transpose(1,0,2)

		if(type_slice=="sagittal"):
			output_3D_current_network=output_3D_current_network.transpose(1,2,0)




		if(perform_postprocessing):
			generate_nii_gz_segmentation_brats2017(output_3D_current_network, path_file_nii_gz)
			output_3D_current_network=read_3D_int_image_nii_gz_brats(path_file_nii_gz)
		else:
			#generate_nii_gz_segmentation_brats2017_without_postprocessing(output_3D_current_network, path_file_nii_gz)

			#generate_nii_gz_segmentation_brats2017_without_postprocessing(output_3D_current_network, path_file_nii_gz)
			generate_nii_gz_segmentation_brats2017_without_postprocessing(output_3D_current_network, path_file_nii_gz)
			#generate_nii_gz_segmentation_august2018(output_3D_current_network, name_file_nii_gz=path_file_nii_gz,path_image_reference=path_image_reference_lungs)




	
	else:
		print("File "+path_file_nii_gz+" already exists")
		if(gt_exists):
			output_3D_current_network=read_3D_int_image_nii_gz_brats(path_file_nii_gz)


	#if GT exists, compute the stats

	if(gt_exists):

		labels_3D_patient=construct_GT_patient(path_gt_axial)
		print("path gt:"+path_gt_axial)

		if(nb_classes==2):
			labels_3D_patient=np.sign(labels_3D_patient)



		if(not(take_all_slices)):
			output_3D_current_network=output_3D_current_network[slice_start_restricted:slice_end_restricted,:,:]
			labels_3D_patient=labels_3D_patient[slice_start_restricted:slice_end_restricted,:,:]



		(precision, recall, dice)=compute_stats_wt_brats_output_one_patient(output_3D_current_network, labels_3D_patient)








		mean_dice_wt=(1.0/float(nb_treated_patients_wt+1))*dice +(float(nb_treated_patients_wt)/float(nb_treated_patients_wt+1))*mean_dice_wt
	
		nb_treated_patients_wt=nb_treated_patients_wt+1

		dices_wt.append(dice)


		if(nb_classes>2):

			(precision_core,recall_core, dice_core, precision_enhancing, recall_enhancing, dice_enhancing)=compute_stats_subclasses_brats_output_one_patient(output_3D_current_network, labels_3D_patient)
			
			if(dice_core>-1.0):
				dices_core.append(dice_core)
				mean_dice_core=(1.0/float(nb_treated_patients_core+1))*dice_core +(float(nb_treated_patients_core)/float(nb_treated_patients_core+1))*mean_dice_core
				nb_treated_patients_core=nb_treated_patients_core+1



			if(dice_enhancing>-1.0):
				dices_enhancing.append(dice_enhancing)
				mean_dice_enhancing=(1.0/float(nb_treated_patients_enhancing+1))*dice_enhancing +(float(nb_treated_patients_enhancing)/float(nb_treated_patients_enhancing+1))*mean_dice_enhancing
				nb_treated_patients_enhancing=nb_treated_patients_enhancing+1



	#break


median_dice_wt=compute_median_from_list(dices_wt)
median_dice_core=compute_median_from_list(dices_core)
median_dice_enhancing=compute_median_from_list(dices_enhancing)





file_dice.write("Mean Dice="+str(mean_dice_wt))
file_dice.write("\nMedian Dice="+str(median_dice_wt))

if(nb_classes>2):
	file_dice.write("\nMean Dice core="+str(mean_dice_core))
	file_dice.write("\nMedian Dice core="+str(median_dice_core))
	file_dice.write("\nMean Dice enhancing="+str(mean_dice_enhancing))
	file_dice.write("\nMedian Dice enhancing="+str(median_dice_enhancing))


file_dice.close()







filename_copy_dice=folder_all_dices+"/"+name_test+"_"+name_net+"_parameters"+str(num_parameters)+".txt"

os.system("cp "+filename_dice+" "+filename_copy_dice)








file_finish=open(filename_finish,"w")

file_finish.write(" ")

file_finish.close()
