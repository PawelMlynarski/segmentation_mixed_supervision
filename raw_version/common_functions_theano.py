import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import os
import sys
#import caffe
import theano
#from pylab import *
#from caffe import layers as L
#from caffe import params as P
import numpy as np
#from common_functions import *
from PIL import Image
import random
import datetime
import h5py

#max_value=300.0
windowing_max=260.0
#windowing_max=300.0





def determine_bounding_boxes(list_files_labels_axial_slices_each_patient):
	#input: for each patient, the filename of the files with labels of axial slices. To access to other slices: replace 'axial' by 'sagittal' or 'coronal'
	#we want each bounding box to be of form [(z1,z2), (y1,y2), (x1,x2)]

	result=[]

	for num_patient in range(len(list_files_labels_axial_slices_each_patient)):
		filename_axial=list_files_labels_axial_slices_each_patient[num_patient]
		filename_sagittal=filename_axial.replace("axial","sagittal")
		filename_coronal=filename_axial.replace("axial","coronal")

		file_labels_axial_slices=open(filename_axial,"r")
		labels_axial_slices=file_labels_axial_slices.read().split("\n")
		labels_axial_slices=[int(j) for j in labels_axial_slices]
		file_labels_axial_slices.close()


		file_labels_sagittal_slices=open(filename_sagittal,"r")
		labels_sagittal_slices=file_labels_sagittal_slices.read().split("\n")
		labels_sagittal_slices=[int(j) for j in labels_sagittal_slices]
		file_labels_sagittal_slices.close()


		file_labels_coronal_slices=open(filename_coronal,"r")
		labels_coronal_slices=file_labels_coronal_slices.read().split("\n")
		labels_coronal_slices=[int(j) for j in labels_coronal_slices]
		file_labels_coronal_slices.close()





		#FAIRE ATTENTION AU CAS OU IL Y A DEUX PARTIES TRES ELOIGNEES DE LA TUMEUR





		nb_axial_slices=len(labels_axial_slices)
		z1=0
		for num_axial_slice in range(nb_axial_slices):
			if(labels_axial_slices[num_axial_slice]>0):
				z1=num_axial_slice
				break

		z2=nb_axial_slices-1
		for a in range(nb_axial_slices):
			num_axial_slice=nb_axial_slices-1-a
			if(labels_axial_slices[num_axial_slice]>0):
				z2=num_axial_slice
				break


		nb_coronal_slices=len(labels_coronal_slices)
		y1=0
		for num_coronal_slice in range(nb_coronal_slices):
			if(labels_coronal_slices[num_coronal_slice]>0):
				y1=num_coronal_slice
				break

		y2=nb_coronal_slices-1
		for a in range(nb_coronal_slices):
			num_coronal_slice=nb_coronal_slices-1-a
			if(labels_coronal_slices[num_coronal_slice]>0):
				y2=num_coronal_slice
				break


		nb_sagittal_slices=len(labels_sagittal_slices)
		x1=0
		for num_sagittal_slice in range(nb_sagittal_slices):
			if(labels_sagittal_slices[num_sagittal_slice]>0):
				x1=num_sagittal_slice
				break

		x2=nb_sagittal_slices-1
		for a in range(nb_sagittal_slices):
			num_sagittal_slice=nb_sagittal_slices-1-a
			if(labels_sagittal_slices[num_sagittal_slice]>0):
				x2=num_sagittal_slice
				break




		result.append([(z1,z2),(y1,y2),(x1,x2)])

	return result



def find_current_parameters(folder_results_one_test):
	files=os.listdir(folder_results_one_test)

	max_parameters=0

	for f in files:
		if("parameters" in f):
			tmp=f.split("parameters")
			if(not(os.path.isdir(folder_results_one_test+"/"+f))):
				continue
			if(len(tmp)<2):
				continue
			#print("f "+f)
			#print(tmp)
			if(tmp[-1]==""):
				continue
				
			num_parameters=int(tmp[-1])

			if(num_parameters>max_parameters):
				max_parameters=num_parameters

	return max_parameters












def copy_conv_parameters_between_networks(net1,net2):

	list_layers_net1=net1.list_layers

	list_layers_net2=net2.list_layers

	if(len(list_layers_net1)!=len(list_layers_net2)):
		print("The 2 networks have a different number of layers")
		sys.exit(1)

	for l in range(len(list_layers_net1)):
		layer_net1=list_layers_net1[l]

		layer_net2=list_layers_net2[l]

		if(layer_net1.name!=layer_net2.name):
			print("something is wrong with the layers (copy of parameters)")
			sys.exit(1)

		if(layer_net1.type=="conv"):
			#print("copying parameters from the layer "+layer_net1.name)
			layer_net2.set_parameters_from_another_layer(layer_net1)
















def find_value_most_recent_folder(folder,common_name):
	#find the maximal value of N in the folders "<common_name>N"
	files=os.listdir(folder)

	max_value=0

	for f in files:
		if(common_name in f):
			tmp=f.split(common_name)
			if(not(os.path.isdir(folder+"/"+f))):
				continue
			if(len(tmp)<2):
				continue
			#print("f "+f)
			#print(tmp)
			value=int(tmp[-1])
			#print(value)

			if(value>max_value):
				max_value=value

	return max_value




def find_layer(name_layer, list_layers):
	result=""
	for k in range(len(list_layers)):
		if(list_layers[k].name==name_layer):
			return list_layers[k]





	print("\n\n\nLAYER NOT FOUND: "+name_layer)
	print("\n\n\n")
	return None




def find_recent_learning_rate(folder_results_one_test):
	#find the most recent file with learning rates, read it and give 
	files=os.listdir(folder_results_one_test)

	max_num_filename=0

	#find the recent file
	file_found=False
	for f in files:
		if("learning_rates" in f):
			file_found=True
			tmp=f.split("learning_rates")
			tmp2=tmp[1].split(".txt")
		

			num_file=int(tmp2[0])

			if(num_file>max_num_filename):
				max_num_filename=num_file



	#read the file (if it exists) and return the value of the current learning rate

	if(not(file_found)):
		print("\n**********************\n\n\n\nDidn't find the file with learning rates**********************\n\n\n\n")
		return -1.0

	filename_learning_rates=folder_results_one_test+"/learning_rates"+str(max_num_filename)+".txt"
	file_learning_rates=open(filename_learning_rates,"r")

	txt_learning_rates=file_learning_rates.read()
	file_learning_rates.close()


	tmp1=txt_learning_rates.split(")]")[0]
	tmp2=tmp1.split(", ")[-1]
	value_learning_rate=float(tmp2)
	print("\n\n\n\nCurrent learning rate: "+str(value_learning_rate)+"\n\n\n\n")
	return value_learning_rate



def find_most_recent_file(path_folder):
	#apply this function for example for the folder "loss" or "dice_class1"
	list_files=os.listdir(path_folder)

	if(len(list_files)==0):
		print("dossier vide:"+path_folder)
		return

	first_filename=list_files[0]

	if(len(list_files)<2):
		print("\n\n\n\n\n\n\n\nfolder "+path_folder+": almost empty")
		return path_folder+"/"+first_filename


	common_extension=first_filename.split(".")[-1]

	common_name=""

	"""
	for k in range(len(first_filename)):
		letter=first_filename[k]
		if(letter.isdigit()):
			break
		else:
			common_name=common_name+letter
	"""
	for k in range(len(first_filename)):
		letter=first_filename[k]
		#if(letter.isdigit()):
			#break

		common_letter=True

		for j in range(len(list_files)):
			if(letter!=list_files[j][k]):
				common_letter=False
				break
		if(common_letter):
			common_name=common_name+letter

		else:
			break
	print("common_name:"+common_name)

	num_max=0

	for filename in list_files:
		tmp=filename.split(common_name)[1].split(".")

			

		if(not(tmp[0].isdigit())):
			continue
		num=int(tmp[0])
		if(num>num_max):
			num_max=num



	path_most_recent_file=path_folder+"/"+common_name+str(num_max)+"."+common_extension

	return path_most_recent_file



def create_lists_coordinates_classes(list_paths_folders_coordinates_classes_each_patient):
	#indexes of the main output: index_patient, class, indexes_voxel_class_one_patient, coordinate (0 for z, 1 for y, 2 for x)
	#other outputs: 
	#list (indexes: num_patient, num_class) of the form [ [number_voxels_class0_patient0,number_voxels_class1_patient0, (...) ], [number_voxels_class0_patient1,number_voxels_class1_patient1, (...) ] ]
	#indexes patients_for_each_class (indexes: class, num_patient_containing_this_class), of the form: [ [indexes patients class0], [indexes patients class1] ]

	print("\n\nReading the coordinates of classes...")
	nb_patients=len(list_paths_folders_coordinates_classes_each_patient)

	nb_classes=len(os.listdir(list_paths_folders_coordinates_classes_each_patient[0]))

	list_coordinates_classes_each_patient=[]
	numbers_voxels_classes_each_patient=[]
	indexes_patients_for_each_class=[]


	for c in range(nb_classes):
		indexes_patients_for_each_class.append([])


	indexes_patients_for_each_class[0]=range(nb_patients)
	
	for num_patient in range(nb_patients):
		list_coordinates_classes_each_patient.append([[]])
		numbers_voxels_classes_each_patient.append([0])

		for c in range(1,nb_classes):
			#list_coordinates_classes_each_patient[num_patient].append([])
			#read the files with coordinates, for this class
			path_file_coordinates_this_class=list_paths_folders_coordinates_classes_each_patient[num_patient]+"/coordinates_class"+str(c)+".txt"
			#print("reading file "+path_file_coordinates_this_class)

			#construct the list of coordinates of voxels of this class in this patient

			f=open(path_file_coordinates_this_class,"r")
			txt=f.read().split("\n")
			f.close()


			coordinates_this_class_this_patient=[]

			for k in range(len(txt)):
				tmp=txt[k].split(" ")
				if(txt[k]==""):
					continue
				#print("tmp")
				#print(tmp)
				coordinates_this_class_this_patient.append([int(tmp[0]),int(tmp[1]),int(tmp[2])])

			list_coordinates_classes_each_patient[num_patient].append(coordinates_this_class_this_patient)
			#remind the number of voxels

			nb_voxels_class_this_patient=len(coordinates_this_class_this_patient)
			numbers_voxels_classes_each_patient[num_patient].append(nb_voxels_class_this_patient)

			#if this number is greater than 0, add the patient to the list of patients containing this class
			if(nb_voxels_class_this_patient>0):
				indexes_patients_for_each_class[c].append(num_patient)


	print("\n\nReading finished")
	return (list_coordinates_classes_each_patient, numbers_voxels_classes_each_patient,indexes_patients_for_each_class)





def generate_hdf5_tensor_feature_maps_one_patient_3D_format(tensor4D,common_path_hdf5):
	#tensor4D, indexes: slice, feature map, y, x

	nb_feature_maps=tensor4D.shape[0]


	for num_feature_map in range(nb_feature_maps):
		tensor3D_one_feature_map=tensor4D[num_feature_map,:,:,:]
		
		#path_result_one_feature_map=common_path_nii_gz+"_fm"+str(num_feature_map)+".nii.gz"
		path_result_one_feature_map=common_path_hdf5.replace("FM0","FM"+str(num_feature_map))

		#os.system("mkdir -p "+path_result_one_feature_map)

		write_tensor_to_hdf5(tensor3D_one_feature_map, path_result_one_feature_map)


def generate_hdf5_tensor_feature_maps_one_patient(tensor4D,common_path_hdf5):
	#tensor4D, indexes: slice, feature map, y, x

	nb_feature_maps=tensor4D.shape[1]


	for num_feature_map in range(nb_feature_maps):
		tensor3D_one_feature_map=tensor4D[:,num_feature_map,:,:]
		
		#path_result_one_feature_map=common_path_nii_gz+"_fm"+str(num_feature_map)+".nii.gz"
		path_result_one_feature_map=common_path_hdf5.replace("FM0","FM"+str(num_feature_map))

		#os.system("mkdir -p "+path_result_one_feature_map)

		write_tensor_to_hdf5(tensor3D_one_feature_map, path_result_one_feature_map)




def write_tensor_to_hdf5(tensor_input, filename_output):

	os.system("rm "+filename_output)
	print("writing to the file "+filename_output)
	f = h5py.File(filename_output, "w")
	f.create_dataset("patient", data=tensor_input, dtype=tensor_input.dtype)
	f.close()





def generate_nii_gz_tensor_feature_maps_one_patient(tensor4D, common_path_nii_gz):


	#tensor4D, indexes: slice, feature map, y, x

	nb_feature_maps=tensor4D.shape[1]


	for num_feature_map in range(nb_feature_maps):
		tensor3D_one_feature_map=tensor4D[:,num_feature_map,:,:]
		
		#path_result_one_feature_map=common_path_nii_gz+"_fm"+str(num_feature_map)+".nii.gz"
		path_result_one_feature_map=common_path_nii_gz.replace("FM0","FM"+str(num_feature_map))

		#os.system("mkdir -p "+path_result_one_feature_map)

		generate_nii_gz_tensor3D_float(tensor3D_one_feature_map, path_result_one_feature_map)






def generate_nii_gz_tensor3D_float(tensor3D, name_file_nii_gz):
	#tensor3D_segmentation: 3D tensor of integers (indexes: z, y, x ))
	
	bin_generate_nii_gz_from_txt="/home/pmlynars/code_Theano2/generation_nii_gz/build/generate_nii_gz_float_given_txt"


		

	#replace the label 3 by the label 4
	tensor_flatten=tensor3D.flatten()
	for k in range(tensor_flatten.shape[0]):
		if(int(tensor_flatten[k])==3):
			#print("replacing 3->4")
			tensor_flatten[k]=4



	#write the result in a txt file
	filename_segm_tmp="/home/pmlynars/code_Theano3/tmp_segm"+str(random.randint(0,100000))+".txt"

	file_segm_txt=open(filename_segm_tmp,"w")
	file_segm_txt.write(" ".join([str(tensor_flatten[k]) for k in range(tensor_flatten.shape[0])]))
	file_segm_txt.close()


	
	print("Generating nii.gz file: "+name_file_nii_gz)
	subprocess.call([bin_generate_nii_gz_from_txt,filename_segm_tmp, name_file_nii_gz])


	os.system("rm "+filename_segm_tmp)





def generate_nii_gz_segmentation_brats2017(tensor3D_segmentation, name_file_nii_gz):
	#tensor3D_segmentation: 3D tensor of integers (indexes: z, y, x ))
	
	bin_generate_nii_gz_from_txt="/home/pmlynars/code_Theano2/generation_nii_gz/build/generate_nii_gz_segmentation_given_txt"
	bin_perform_postprocessing="/home/pmlynars/code_Theano2/generation_nii_gz/build/postprocess_segmentation"
	





	bin_ants_thresholding="/home/pmlynars/Programmes/ANTs/build/bin/ThresholdImage"
	bin_ants_stats_components="/home/pmlynars/Programmes/ANTs/build/bin/GetConnectedComponentsFeatureImages"






	#replace the label 3 by the label 4
	tensor_flatten=tensor3D_segmentation.flatten()
	for k in range(tensor_flatten.shape[0]):
		if(int(tensor_flatten[k])==3):
			#print("replacing 3->4")
			tensor_flatten[k]=4



	#write the result in a txt file
	filename_segm_tmp="/home/pmlynars/code_Theano2/tmp_segm"+str(random.randint(0,100000))+".txt"

	file_segm_txt=open(filename_segm_tmp,"w")
	file_segm_txt.write(" ".join([str(tensor_flatten[k]) for k in range(tensor_flatten.shape[0])]))
	file_segm_txt.close()



	name_patient=(name_file_nii_gz.split("/")[-1]).split(".")[0]
	folder_images="/".join(name_file_nii_gz.split("/")[0:-1])


	folder_stats_components=folder_images+"/stats_components/"
	os.system("mkdir -p "+folder_stats_components)

	


	#np.savetxt(file_segm_tmp, tensor_flatten)
	name_file_nii_gz_before_postprocessing=folder_stats_components+"/"+name_patient+"_before_postprocessing.nii.gz"
	
	print("Generating nii.gz file")
	subprocess.call([bin_generate_nii_gz_from_txt,filename_segm_tmp, name_file_nii_gz_before_postprocessing])


	


	#generate a binary mask
	print("Creating a binary mask")
	name_file_mask=folder_stats_components+"/"+name_patient+"_binary.nii.gz"
	subprocess.call([bin_ants_thresholding,"3", name_file_nii_gz_before_postprocessing, name_file_mask,"0.9","5.0"])


	print("\n\n\nAnalysis of shapes")
	prefix_outputs_ants=folder_stats_components+"/"+name_patient
	subprocess.call([bin_ants_stats_components,"3",name_file_mask,prefix_outputs_ants])

	filename_image_volume=prefix_outputs_ants+"PHYSICAL_VOLUME.nii.gz"
	filename_image_ratio=prefix_outputs_ants+"VOLUME_TO_SURFACE_AREA_RATIO.nii.gz"
	filename_image_elongation=prefix_outputs_ants+"ELONGATION.nii.gz"
	filename_image_eccentricity=prefix_outputs_ants+"ECCENTRICITY.nii.gz"

	print("\n\n\nPostprocessing")
	#fileSegmentation_nii_gz fileVolume_nii_gz fileRatioVolumeSurface_nii_gz fileElongation_nii_gz fileEccentricity_nii_gz fileOutput_nii_gz
	subprocess.call([bin_perform_postprocessing, name_file_nii_gz_before_postprocessing, filename_image_volume, filename_image_ratio, filename_image_elongation, filename_image_eccentricity,name_file_nii_gz])


	os.system("rm "+filename_segm_tmp)







def generate_nii_gz_segmentation_brats2017_without_postprocessing(tensor3D_segmentation, name_file_nii_gz):
	#tensor3D_segmentation: 3D tensor of integers (indexes: z, y, x ))
	
	bin_generate_nii_gz_from_txt="/home/pmlynars/code_Theano2/generation_nii_gz/build/generate_nii_gz_segmentation_given_txt"





	#replace the label 3 by the label 4
	tensor_flatten=tensor3D_segmentation.flatten()
	for k in range(tensor_flatten.shape[0]):
		if(int(tensor_flatten[k])==3):
			#print("replacing 3->4")
			tensor_flatten[k]=4



	#write the result in a txt file
	filename_segm_tmp="/home/pmlynars/code_Theano2/tmp_segm"+str(random.randint(0,100000))+".txt"

	file_segm_txt=open(filename_segm_tmp,"w")
	file_segm_txt.write(" ".join([str(tensor_flatten[k]) for k in range(tensor_flatten.shape[0])]))
	file_segm_txt.close()



	name_patient=(name_file_nii_gz.split("/")[-1]).split(".")[0]
	folder_images="/".join(name_file_nii_gz.split("/")[0:-1])

	


	#np.savetxt(file_segm_tmp, tensor_flatten)
	name_file_nii_gz_before_postprocessing=folder_images+"/"+name_patient+".nii.gz"
	
	print("Generating nii.gz file")
	subprocess.call([bin_generate_nii_gz_from_txt,filename_segm_tmp, name_file_nii_gz_before_postprocessing])


	os.system("rm "+filename_segm_tmp)



def read_3D_int_image_nii_gz_brats(path_file_nii_gz):
	#result: 3D tensor (indexes: [z,y,x])
	
	bin_extract_axial_slices_npy="/home/pmlynars/extraction_slices/build/extractAxialSlicesTxt"
	folder_tmp_slices="folders_tmp_slices/"+path_file_nii_gz.split("/")[-1]+str(random.randint(0,1000000))

	os.system("mkdir -p "+folder_tmp_slices)
	#extact .txt slices
	subprocess.call([bin_extract_axial_slices_npy, path_file_nii_gz, folder_tmp_slices])


	#determine the dimensions of the output
	filenames_txt=os.listdir(folder_tmp_slices)
	nb_axial_slices=len(filenames_txt)


	first_image=np.loadtxt(folder_tmp_slices+"/"+filenames_txt[0])

	dim_y=first_image.shape[0]
	dim_x=first_image.shape[1]

	result=np.zeros([nb_axial_slices,dim_y,dim_x],dtype=np.int32)
	#read files

	for s in range(nb_axial_slices):
		#result[s,:,:]=np.loadtxt(folder_tmp_slices+"/"+filenames_txt[s])
		result[s,:,:]=np.loadtxt(folder_tmp_slices+"/slice"+str(s)+".txt")
		
		"""
		if(s==89):
			print("result["+str(s)+"]")
			#print(result[s,145:165,75:85])
			print(result[s,100:115,100:115])
	
		"""
		

		#replace the label 4 by the label 3
		result[result==4]=3


	#clean
	os.system("rm -r "+folder_tmp_slices)


	return result


def generate_nii_gz_segmentation_brats2017_old(tensor3D_segmentation, name_file_nii_gz):
	#tensor3D_segmentation: 3D tensor of integers (indexes: z, y, x ))
	
	#write the slices in the files and 
	filename_segm_tmp="/home/pmlynars/code_Theano2/tmp_segm"+str(random.randint(0,100000))+".txt"
	

	#replace the label 3 by the label 4
	tensor_flatten=tensor3D_segmentation.flatten()
	for k in range(tensor_flatten.shape[0]):
		if(int(tensor_flatten[k])==3):
			#print("replacing 3->4")
			tensor_flatten[k]=4


	file_segm_txt=open(filename_segm_tmp,"w")
	file_segm_txt.write(" ".join([str(tensor_flatten[k]) for k in range(tensor_flatten.shape[0])]))
	file_segm_txt.close()
	#np.savetxt(file_segm_tmp, tensor_flatten)
	print("Generating nii.gz file")
	subprocess.call(["generation_nii_gz/build/generate_nii_gz_segmentation_given_txt",filename_segm_tmp, name_file_nii_gz])
	os.system("rm "+filename_segm_tmp)
	


def compute_stats_all_subclasses_output_tensor_labels(output_labels, gt_labels, nb_classes):
	#inputs: 3D or 4D tensors of integers: dimensions are (dims: nb_images,<z>,y,x)
	#result: list of the form [(precision_class0, recall_class0, dice_class0),..., (precision_class_K, recall_class_K, dice_class_K)]
	result=[]

	for cl in range(nb_classes):
		examples_labeled_as_positive_class=(output_labels==cl)
		#not core: edema or healthy
		examples_labeled_as_negative_class=(output_labels!=cl)
		

		positive_examples_class=(gt_labels==cl)
		negative_examples_class=(gt_labels!=cl)

		#(precision_class, recall_class, dice_class)=compute_stats_two_masks(examples_labeled_as_positive_class,examples_labeled_as_negative_class, positive_examples_class, negative_examples_class)
		#print("\n\nClass "+str(cl)+":")
		tuple_stats_class=compute_stats_two_masks_list(examples_labeled_as_positive_class,examples_labeled_as_negative_class, positive_examples_class, negative_examples_class)
		result.append(tuple_stats_class)


	return result










def compute_norm_gradient(list_gradients):
	#list_gradients: as in 'train_net.py'
	result=0.0

	nb_el=len(list_gradients)

	for j in range(nb_el):
		result=result+np.sum(list_gradients[j]**2)
	
	result=np.sqrt(result)
	
	return result







def compute_norm_gradient_all_layers(list_gradients):
	#list_gradients: as in 'train_net.py'
	result=[]

	nb_el=len(list_gradients)

	for j in range(nb_el):
		result.append(np.sqrt(np.sum(list_gradients[j]**2)))
	
	
	return result




def compute_stats_two_masks_list(examples_labeled_as_positive,examples_labeled_as_negative, positive_examples, negative_examples):
	#each input= 3D integer tensor with ones and zeros
	true_positives=examples_labeled_as_positive*positive_examples
	false_positives=examples_labeled_as_positive*negative_examples
	false_negatives=examples_labeled_as_negative*positive_examples

	nb_true_positives=np.sum(np.sum(np.sum(true_positives)))
	nb_false_positives=np.sum(np.sum(np.sum(false_positives)))
	nb_false_negatives=np.sum(np.sum(np.sum(false_negatives)))


	nb_positive_examples=np.sum(np.sum(np.sum(positive_examples)))
	nb_examples_labeled_as_positive=nb_true_positives+nb_false_positives


	#print("\n\nThere are "+str(nb_true_positives+nb_false_negatives)+" positive examples")
	#print("There are "+str(nb_true_positives+nb_false_positives)+" examples predicted as positive")


	if((nb_true_positives+nb_false_negatives)>0):
		recall=float(nb_true_positives)/float(nb_true_positives+nb_false_negatives)
	else:
		recall=-1.0

	if((nb_true_positives+nb_false_positives)>0):
		precision=float(nb_true_positives)/float(nb_true_positives+nb_false_positives)
	else:
		precision=-1.0

	if((nb_positive_examples+nb_examples_labeled_as_positive)>0):
		dice=float(2*nb_true_positives)/float(nb_positive_examples+nb_examples_labeled_as_positive)
	else:
		dice=-1.0

	return [precision, recall, dice]







def compute_dice_given_numbers_of_occurrences(nb_positive_examples,nb_true_positives ,nb_examples_labeled_as_positive):
	if((nb_positive_examples+nb_examples_labeled_as_positive)>0):
		result=float(2*nb_true_positives)/float(nb_positive_examples+nb_examples_labeled_as_positive)
	else:
		result=-1.0

	return result


def compute_positives_and_negatives_occurrences_two_masks(examples_labeled_as_positive,examples_labeled_as_negative, positive_examples, negative_examples):
	#each input= 3D integer tensor with ones and zeros
	#compute the stats ncessary for computing Dice coefficient

	true_positives=examples_labeled_as_positive*positive_examples
	false_positives=examples_labeled_as_positive*negative_examples
	false_negatives=examples_labeled_as_negative*positive_examples

	nb_true_positives=np.sum(np.sum(np.sum(true_positives)))
	nb_false_positives=np.sum(np.sum(np.sum(false_positives)))
	nb_false_negatives=np.sum(np.sum(np.sum(false_negatives)))


	nb_positive_examples=np.sum(np.sum(np.sum(positive_examples)))
	nb_examples_labeled_as_positive=nb_true_positives+nb_false_positives


	return [nb_positive_examples, nb_true_positives, nb_examples_labeled_as_positive]


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


	#print("\n\nThere are "+str(nb_true_positives+nb_false_negatives)+" positive examples")
	#print("There are "+str(nb_true_positives+nb_false_positives)+" examples predicted as positive")


	if((nb_true_positives+nb_false_negatives)>0):
		recall=float(nb_true_positives)/float(nb_true_positives+nb_false_negatives)
	else:
		recall=-1.0

	if((nb_true_positives+nb_false_positives)>0):
		precision=float(nb_true_positives)/float(nb_true_positives+nb_false_positives)
	else:
		precision=-1.0

	if((nb_positive_examples+nb_examples_labeled_as_positive)>0):
		dice=float(2*nb_true_positives)/float(nb_positive_examples+nb_examples_labeled_as_positive)
	else:
		dice=-1.0

	return (precision, recall, dice)



def compute_recall_and_precision(output_labels, target_labels):
	#inputs: 3D or 4D tensors of integers
	
	examples_labeled_as_positive=(output_labels>0)
	examples_labeled_as_negative=(output_labels==0)
	positive_examples=(target_labels>0)
	negative_examples=(target_labels==0)





	true_positives=examples_labeled_as_positive*positive_examples
	false_positives=examples_labeled_as_positive*negative_examples
	false_negatives=examples_labeled_as_negative*positive_examples

	nb_true_positives=np.sum(np.sum(np.sum(true_positives)))
	nb_false_positives=np.sum(np.sum(np.sum(false_positives)))
	nb_false_negatives=np.sum(np.sum(np.sum(false_negatives)))



	#print("\n\nThere are "+str(nb_true_positives+nb_false_negatives)+" positive examples")
	#print("There are "+str(nb_true_positives+nb_false_positives)+" examples predicted as positive")

	if((nb_true_positives+nb_false_negatives)>0):
		recall=float(nb_true_positives)/float(nb_true_positives+nb_false_negatives)
	else:
		recall=0.0

	if((nb_true_positives+nb_false_positives)>0):
		precision=float(nb_true_positives)/float(nb_true_positives+nb_false_positives)
	else:
		precision=0.0

	return (recall, precision)




def determine_bounding_boxes(list_files_labels_axial_slices_each_patient):
	#input: for each patient, the filename of the files with labels of axial slices. To access to other slices: replace 'axial' by 'sagittal' or 'coronal'
	#we want each bounding box to be of form [(z1,z2), (y1,y2), (x1,x2)]

	result=[]

	for num_patient in range(len(list_files_labels_axial_slices_each_patient)):
		filename_axial=list_files_labels_axial_slices_each_patient[num_patient]
		filename_sagittal=filename_axial.replace("axial","sagittal")
		filename_coronal=filename_axial.replace("axial","coronal")

		file_labels_axial_slices=open(filename_axial,"r")
		labels_axial_slices=file_labels_axial_slices.read().split("\n")
		labels_axial_slices=[int(j) for j in labels_axial_slices]
		file_labels_axial_slices.close()


		file_labels_sagittal_slices=open(filename_sagittal,"r")
		labels_sagittal_slices=file_labels_sagittal_slices.read().split("\n")
		labels_sagittal_slices=[int(j) for j in labels_sagittal_slices]
		file_labels_sagittal_slices.close()


		file_labels_coronal_slices=open(filename_coronal,"r")
		labels_coronal_slices=file_labels_coronal_slices.read().split("\n")
		labels_coronal_slices=[int(j) for j in labels_coronal_slices]
		file_labels_coronal_slices.close()





		#FAIRE ATTENTION AU CAS OU IL Y A DEUX PARTIES TRES ELOIGNEES DE LA TUMEUR





		nb_axial_slices=len(labels_axial_slices)
		z1=0
		for num_axial_slice in range(nb_axial_slices):
			if(labels_axial_slices[num_axial_slice]>0):
				z1=num_axial_slice
				break

		z2=nb_axial_slices-1
		for a in range(nb_axial_slices):
			num_axial_slice=nb_axial_slices-1-a
			if(labels_axial_slices[num_axial_slice]>0):
				z2=num_axial_slice
				break


		nb_coronal_slices=len(labels_coronal_slices)
		y1=0
		for num_coronal_slice in range(nb_coronal_slices):
			if(labels_coronal_slices[num_coronal_slice]>0):
				y1=num_coronal_slice
				break

		y2=nb_coronal_slices-1
		for a in range(nb_coronal_slices):
			num_coronal_slice=nb_coronal_slices-1-a
			if(labels_coronal_slices[num_coronal_slice]>0):
				y2=num_coronal_slice
				break


		nb_sagittal_slices=len(labels_sagittal_slices)
		x1=0
		for num_sagittal_slice in range(nb_sagittal_slices):
			if(labels_sagittal_slices[num_sagittal_slice]>0):
				x1=num_sagittal_slice
				break

		x2=nb_sagittal_slices-1
		for a in range(nb_sagittal_slices):
			num_sagittal_slice=nb_sagittal_slices-1-a
			if(labels_sagittal_slices[num_sagittal_slice]>0):
				x2=num_sagittal_slice
				break




		result.append([(z1,z2),(y1,y2),(x1,x2)])

	return result





def combine_2channels_gradient(gradient):
	return np.maximum(abs(gradient[0,:,:]),abs(gradient[1,:,:]))


def combine_4channels_gradient(gradient):
	return np.maximum(abs(gradient[0,:,:]),np.maximum(abs(gradient[3,:,:]),np.maximum(abs(gradient[1,:,:]),abs(gradient[2,:,:]))))




def show_evolution(evo,result_filename):
	fig=plt.figure()
	plt.plot(range(len(evo)),evo)
	#plt.show()
	plt.savefig(result_filename)
	plt.close(fig)
	#f.close()


def visualize_matrix(gradient_matrix_form,result_filename_image):
	#a=gradient_matrix_form[0,0,:,:]
	#print(a.shape)

	fig=plt.figure()
	#plt.imshow(gradient_matrix_form,cmap='Greys_r')
	plt.imshow(gradient_matrix_form,cmap='spectral')
	#print(a[80][80])
	plt.savefig(result_filename_image)
	plt.close(fig)


def add_zero_padding(tensor3,target_shape):


	#FLOAT??



	result=np.zeros(target_shape,dtype=np.int32)
	#result=np.zeros(target_shape,dtype=theano.config.floatX)
	offset_y=(target_shape[1]-tensor3.shape[1])/2
	offset_x=(target_shape[2]-tensor3.shape[2])/2
	result[:,offset_y:(offset_y+tensor3.shape[1]),offset_x:(offset_x+tensor3.shape[2])]=tensor3
	return result


def add_zero_padding_tensor4(tensor4,target_shape):
	result=np.zeros(target_shape,dtype=theano.config.floatX)
	offset_y=(target_shape[2]-tensor4.shape[2])/2
	offset_x=(target_shape[3]-tensor4.shape[3])/2
	result[:,:,offset_y:(offset_y+tensor4.shape[2]),offset_x:(offset_x+tensor4.shape[3])]=tensor4
	return result


def add_zero_padding_3D(tensor5,target_shape):
	result=np.zeros(target_shape,dtype=np.int32)
	offset_z=(target_shape[1]-tensor5.shape[1])/2
	offset_y=(target_shape[2]-tensor5.shape[2])/2
	offset_x=(target_shape[3]-tensor5.shape[3])/2
	result[:,offset_z:(offset_z+tensor5.shape[1]),offset_y:(offset_y+tensor5.shape[2]),offset_x:(offset_x+tensor5.shape[3])]=tensor5
	return result


def adjust_shape_tensor3(tensor3,target_shape,nb_zeros_top_nn, nb_zeros_left_nn, offset_y_nn,offset_x_nn):
	#target_shape= shape of the 3D tensor with voxelwise labels
	#tensor3= 3D tensor with output labels of a neural net which can have:
	#1) bigger dimensions if too many zeros were added to handle the borders)
	#2) smaller dimensions if not enough zeros were added
	result=np.zeros(target_shape,dtype=np.int32)

	diff_y=tensor3.shape[1]-target_shape[1]
	diff_x=tensor3.shape[2]-target_shape[2]


	#PAS EXACT MAIS DEVRAIT MARCHER DANS LA PLUPART DE CAS



	if(diff_y<0):
		#input too small: "add zeros"
		start_y_input_tensor3=0
		start_y_output_tensor3=offset_y_nn-nb_zeros_top_nn

		#'-nb_zeros_top_nn' because the second tensor wasn't padded with zeros!
		nb_values_y=tensor3.shape[1]

	else:
		#"crop" the input
		start_y_input_tensor3=nb_zeros_top_nn-offset_y_nn
		start_y_output_tensor3=0
		nb_values_y=target_shape[1]


	if(diff_x<0):
		#input too small: "add zeros"
		start_x_input_tensor3=0
		start_x_output_tensor3=offset_x_nn-nb_zeros_left_nn
		nb_values_x=tensor3.shape[2]
	else:
		#"crop" the input
		start_x_input_tensor3=nb_zeros_left_nn-offset_x_nn
		start_x_output_tensor3=0
		nb_values_x=target_shape[2]


	result[:,start_y_output_tensor3:(start_y_output_tensor3+nb_values_y),start_x_output_tensor3:(start_x_output_tensor3+nb_values_x)]=tensor3[:,start_y_input_tensor3:(start_y_input_tensor3+nb_values_y),start_x_input_tensor3:(start_x_input_tensor3+nb_values_x)]
	return result








def adjust_shape_tensor4(tensor4,target_shape_tensor4,nb_zeros_top_nn, nb_zeros_left_nn, offset_y,offset_x):
	#ATTENTION: "offset" is not necessiraly the total offset of the network
	result=np.zeros(target_shape_tensor4,dtype=theano.config.floatX)

	"""
	print("SHAPE INPUT TENSOR4")
	print(tensor4.shape)

	print("TARGET SHAPE TENSOR4")
	print(target_shape_tensor4)

	print("parameters function")
	print(nb_zeros_top_nn)
	print(nb_zeros_left_nn)
	print(offset_y)
	print(offset_x)
	"""
	diff_y=tensor4.shape[2]-target_shape_tensor4[2]
	diff_x=tensor4.shape[3]-target_shape_tensor4[3]





	#A VERIFIER


	if(nb_zeros_top_nn<=offset_y):
		#we added not enough zeros to the input of the network
		#probably need to 'pad zeros' to the input
		start_y_input_tensor4=0
		start_y_output_tensor4=offset_y-nb_zeros_top_nn
		nb_values_y=min(tensor4.shape[2],target_shape_tensor4[2]-start_y_output_tensor4)
	else:
		#we added too many zeros to the input of the network
		#probably need to crop the input
		start_y_input_tensor4=nb_zeros_top_nn-offset_y
		start_y_output_tensor4=0
		nb_values_y=min(tensor4.shape[2]-start_y_input_tensor4,target_shape_tensor4[2])


	if(nb_zeros_left_nn<=offset_x):
		#we added not enough zeros to the input of the network
		#probably need to 'pad zeros' to the input
		start_x_input_tensor4=0
		start_x_output_tensor4=offset_x-nb_zeros_left_nn
		nb_values_x=min(tensor4.shape[3],target_shape_tensor4[3]-start_x_output_tensor4)
	else:
		#we added too many zeros to the input of the network
		#probably need to crop the input
		start_x_input_tensor4=nb_zeros_left_nn-offset_x
		start_x_output_tensor4=0
		nb_values_x=min(tensor4.shape[3]-start_x_input_tensor4,target_shape_tensor4[3])


	result[:,:,start_y_output_tensor4:(start_y_output_tensor4+nb_values_y),start_x_output_tensor4:(start_x_output_tensor4+nb_values_x)]=tensor4[:,:,start_y_input_tensor4:(start_y_input_tensor4+nb_values_y),start_x_input_tensor4:(start_x_input_tensor4+nb_values_x)]
	return result



"""
def adjust_shape_tensor4_old(tensor4,target_shape_tensor4,nb_zeros_top_nn, nb_zeros_left_nn, offset_y,offset_x):
	#ATTENTION: "offset" is not necessiraly the total offset of the network
	result=np.zeros(target_shape_tensor4,dtype=theano.config.floatX)


	print("SHAPE INPUT TENSOR4")
	print(tensor4.shape)

	print("TARGET SHAPE TENSOR4")
	print(target_shape_tensor4)

	print("parameters function")
	print(nb_zeros_top_nn)
	print(nb_zeros_left_nn)
	print(offset_y)
	print(offset_x)

	diff_y=tensor4.shape[2]-target_shape_tensor4[2]
	diff_x=tensor4.shape[3]-target_shape_tensor4[3]



	if(diff_y<0):
		#input too small: "add zeros"
		start_y_input_tensor4=0
		start_y_output_tensor4=offset_y-nb_zeros_top_nn
		nb_values_y=tensor4.shape[2]

	else:
		#"crop" the input
		start_y_input_tensor4=nb_zeros_top_nn-offset_y
		start_y_output_tensor4=0
		nb_values_y=target_shape_tensor4[2]


	if(diff_x<0):
		#input too small: "add zeros"
		start_x_input_tensor4=0
		start_x_output_tensor4=offset_x-nb_zeros_left_nn
		nb_values_x=tensor4.shape[3]
	else:
		#"crop" the input
		start_x_input_tensor4=nb_zeros_left_nn-offset_x
		start_x_output_tensor4=0
		nb_values_x=target_shape_tensor4[3]


	result[:,:,start_y_output_tensor4:(start_y_output_tensor4+nb_values_y),start_x_output_tensor4:(start_x_output_tensor4+nb_values_x)]=tensor4[:,:,start_y_input_tensor4:(start_y_input_tensor4+nb_values_y),start_x_input_tensor4:(start_x_input_tensor4+nb_values_x)]
	return result

"""





def create_tensor_input_tensor_labels_from_one_list(list_all_images_precised_label, list_indexes_start, shape_batch, list_modalities):
	#list_all_images_precised_label=[(path_slice_0_flair,label_slice), ...]
	#labels indicate if we have to load the GT or leave it at zeros
	
	nb_modalities=len(list_modalities)

	nb_total_images=len(list_all_images_precised_label)

	batch_size=shape_batch[0]
	height=shape_batch[2]
	width=shape_batch[3]

	#initialize tensor
	tensor_images=np.zeros(shape_batch,dtype=theano.config.floatX)
	

	tensor_labels=np.zeros([batch_size,shape_batch[2],shape_batch[3]],dtype=np.int32)


	#index for filling the tensors
	num_image=-1


	#Attention: the order of extraction has an importance. We don't take the GT from the last images
	#Remark: if an image is labeled 0, do not read the GT

	for k in range(batch_size):
		num_image=num_image+1

		index=list_indexes_start[0]

		#check if we need to reinitialize the list (put index to 0 and shuffle the list)
		if(index==nb_total_images):
			list_indexes_start[0]=0
			random.shuffle(lists_images_training[0])
			index=0

		#print("je vais charger "+lists_images_training[0][index])
		path_flair=list_all_images_precised_label[index][0]
		label=list_all_images_precised_label[index][1]

		for mod in range(nb_modalities):	
			modality=list_modalities[mod]
			path_one_modality=path_flair.replace("T2_FLAIR",modality)
			tensor_images[num_image, mod, :, :]=np.load(path_one_modality)
			

		if(label==1):
			tensor_labels[num_image,:,:]=np.load(path_flair.replace("T2_FLAIR","GT"))
			#print("\n\n\n\n\nje charge la GT de "+path_flair.replace("T2_FLAIR","GT"))
		
		#otherwise don't read the GT: tensor_labels[num_image,:,:] is already at zeros

		#Increment the start index
		list_indexes_start[0]=list_indexes_start[0]+1

	

	if(num_image!=(batch_size-1)):
		print("\n\n\n\nfonction 'create_tensor': THERE WAS A PROBLEM WITH A NUMBER OF EXTRACTED IMAGES\n\n\n\n")
	

	#looks correct

	return (tensor_images,tensor_labels)
	











def create_lists_paths_slices_and_labels_slices(filename_file_paths_folders_slices_flair):
	#file_folders_flair_patients_training_with_gt=open("lists_images/miccai_training_patients_flair_axial_with_gt.txt","r")
	file_folders_flair_patients=open(filename_file_paths_folders_slices_flair,"r")

	list_folders_flair_patients=file_folders_flair_patients.read().split("\n")
	file_folders_flair_patients.close()


	list_files_labels_each_patient_with_gt=[]

	for folder_flair_patient in list_folders_flair_patients:
		list_files_labels_each_patient_with_gt.append(folder_flair_patient.replace("slices_bin_normalized_without_registration/T2_FLAIR/","labels_slices_normalized_without_registration/")+".txt")

	return (list_folders_flair_patients,list_files_labels_each_patient_with_gt)




def visualize_matrix_gray(matrix,result_filename_image):
	#a=gradient_matrix_form[0,0,:,:]
	#print(a.shape)






	#windowing
	#matrix[0,0]=0
	#matrix[0,1]=500

	fig=plt.figure()
	plt.imshow(matrix,cmap='Greys_r')
	#plt.imshow(gradient_matrix_form,cmap='spectral')
	#print(a[80][80])
	plt.savefig(result_filename_image)
	plt.close(fig)

def visualize_segmentation(matrix_segmentation,result_filename_image):
	#we assume we have the classes 0, 1, 2, 3, 4. We want the last one to be white
	matrix_colors=matrix_segmentation*51
	matrix_colors[0,0]=254 #for the windowing

	fig=plt.figure()
	plt.imshow(matrix_colors,cmap='Greys_r')
	#plt.imshow(gradient_matrix_form,cmap='spectral')
	#print(a[80][80])
	plt.savefig(result_filename_image)
	plt.close(fig)



def visualize_segmentation_4classes(matrix_segmentation,result_filename_image):
	#we assume that we have the classes 0, 1, 2, 3. We want the last one to be white
	matrix_colors=matrix_segmentation*80
	matrix_colors[0,0]=254 #for the windowing

	fig=plt.figure()
	plt.imshow(matrix_colors,cmap='Greys_r')
	#plt.imshow(gradient_matrix_form,cmap='spectral')
	#print(a[80][80])
	plt.savefig(result_filename_image)
	plt.close(fig)




def display_matrix(matrix):
	height=matrix.shape[0]
	width=matrix.shape[1]

	for y in range(height):
		for x in range(width):
			sys.stdout.write(str('%.2f' % matrix[y][x])+" ")

			#sys.stdout.write((str(int(round(matrix[y][x])))+" "))
		print("")
		#print(matrix[y,:])


def visualize_3D_segmentation(tensor3D, folder_result, slice_start):
	print("visualizying 3D segmentation...")
	recursive_mkdir(folder_result)
	nb_slices=tensor3D.shape[0]
	for s in range(nb_slices):
		filename_slice=folder_result+"/slice"+str(s+slice_start)+".png"
		visualize_segmentation(tensor3D[s,:,:],filename_slice)
	print("done")


def visualize_3D_tensor_given_path_folder(path_folder_flair_axial_patient, folder_result, slice_start, slice_end):
	print("visualizying 3D tensor...")
	#for displaying 3D input data
	recursive_mkdir(folder_result)
	#nb_slices=tensor3D.shape[0]
	for s in range(slice_start, slice_end+1):
		filename_slice_npy=path_folder_flair_axial_patient+"/slice"+str(s)+".npy"
		slice_npy=np.load(filename_slice_npy)
		filename_slice_png=folder_result+"/slice"+str(s)+".png"
		visualize_matrix_gray(slice_npy,filename_slice_png)
	print("done")


def visualize_3D_segmentation_color(path_folder_flair_axial_patient, tensor3D_labels, folder_result, slice_start, color):
	print("visualizying 3D segmentation...")
	recursive_mkdir(folder_result)
	nb_slices=tensor3D_labels.shape[0]
	for s in range(nb_slices):
		num_slice=s+slice_start
		filename_slice_npy=path_folder_flair_axial_patient+"/slice"+str(num_slice)+".npy"
		slice_npy=np.load(filename_slice_npy)
		slice_segm=tensor3D_labels[s,:,:]

		filename_slice_result=folder_result+"/slice"+str(num_slice)+".png"
		visualize_segmentation_color(slice_npy,slice_segm,color,filename_slice_result)
		#visualize_segmentation_contour(slice_npy,slice_segm,filename_slice_result)
	print("done")


#def visualize_segmentation_contour(slice_image_float_npy,slice_segm_int_npy,filename_slice_result):


def visualize_segmentation_color(input_np_2d_float, segmentation_2D_np_int, color, filename_result):
	height=input_np_2d_float.shape[0]
 	width=input_np_2d_float.shape[1]



	#img_result=Image.new("RGBA",(width,height),(0,0,0,255))
	

	#we want float values between 0 and 1 and in RGBA format

	max_value=windowing_max
	min_value=0.0


	input_correct_ranges=(np.minimum(input_np_2d_float/max_value,1.0)*255.0)

	
	input_rgb=Image.new("RGBA",(width,height),(0,0,0))

	for y in range(height):
		for x in range(width):
			value=int(input_correct_ranges[y,x])

			#input_rgba.putpixel((x,y),(value,value,value,255))
			input_rgb.putpixel((x,y),(value,value,value))


	segm_rgba=Image.new("RGBA",(width,height),(0,0,0,255))
	segm_rgb=Image.new("RGBA",(width,height),(0,0,0))
	"""
	for y in range(height):
		for x in range(width):
			segm_rgba.putpixel((x,y),input_rgba.getpixel((x,y)))

	###a checker si le max est bien applique
	"""

	for y in range(height):
		for x in range(width):
			#segm_rgba[:,:,0]=(segmentation_2D_np_int/np.max(segmentation_2D_np_int))*255
			#print(value)

			value=int((segmentation_2D_np_int[y,x]/np.max(segmentation_2D_np_int))*255)
			if(value>0):
				if(color=="red"):
					segm_rgba.putpixel((x,y),(value,0,0,200))
					segm_rgb.putpixel((x,y),(value,0,0))
				if(color=="green"):
					segm_rgba.putpixel((x,y),(0, value,0,200))
					segm_rgb.putpixel((x,y),(0,value,0))
				if(color=="blue"):
					segm_rgba.putpixel((x,y),(0,0,value,200))
					segm_rgb.putpixel((x,y),(0,0,value))



	#input_rgba.paste(segm_rgba,(0,0),segm_rgba)
	
	#print("j'enregistre l'image dans "+filename_result)
	img_result=Image.composite(input_rgb,segm_rgb,segm_rgba)
	
	img_result.save(filename_result,format='png')
	#input_rgba.save(filename_result,format='png')





def visualize_3D_segmentation_contour(path_folder_flair_axial_patient, tensor3D_labels, folder_result, slice_start, color):
	print("visualizying 3D segmentation...")
	recursive_mkdir(folder_result)
	nb_slices=tensor3D_labels.shape[0]
	for s in range(nb_slices):
		num_slice=s+slice_start
		filename_slice_npy=path_folder_flair_axial_patient+"/slice"+str(num_slice)+".npy"
		slice_npy=np.load(filename_slice_npy)
		slice_segm=tensor3D_labels[s,:,:]

		filename_slice_result=folder_result+"/slice"+str(num_slice)+".png"
		visualize_segmentation_contour(slice_npy,slice_segm,color,filename_slice_result)
		#visualize_segmentation_contour(slice_npy,slice_segm,filename_slice_result)
	print("done")


def visualize_3D_image(path_folder_flair_axial_patient, folder_result, slice_start, slice_end):
	print("visualizying 3D image...")
	recursive_mkdir(folder_result)

	for s in range(slice_start, slice_end+1):
		num_slice=s
		filename_slice_npy=path_folder_flair_axial_patient+"/slice"+str(num_slice)+".npy"
		slice_npy=np.load(filename_slice_npy)
		

		filename_slice_result=folder_result+"/slice"+str(num_slice)+".png"
		visualize_slice(slice_npy,filename_slice_result)
		#visualize_segmentation_contour(slice_npy,slice_segm,filename_slice_result)
	print("done")



#def visualize_segmentation_contour(slice_image_float_npy,slice_segm_int_npy,filename_slice_result):










def show_training_batch_multiscale_3D(list_patches_3D_multiscale,list_labels_3D_multiscale, folder_output):


	nb_images=list_patches_3D_multiscale[0].shape[0]
	nb_scales=len(list_patches_3D_multiscale)


	for num_image in range(nb_images):
		
		one_image_multiscale=[]
		one_image_labels_multiscale=[]

		for s in range(nb_scales):
			one_image_multiscale.append(list_patches_3D_multiscale[s][num_image])
			one_image_labels_multiscale.append(list_labels_3D_multiscale[s][num_image])


		folder_output_one_image=folder_output+"/image"+str(num_image)
		visualize_multiscale_patch_3D(one_image_multiscale,folder_output_one_image)
		visualize_labels_multiscale_patch_3D(one_image_labels_multiscale,folder_output_one_image)



def visualize_multiscale_patch_3D(list_tensors_4D,folder_output):
	#list_tensors_4D= [tensor4D_image_scale1, tensor4D_image_scale2, tensor4D_image_scale3]

	nb_scales=len(list_tensors_4D)
	#nb_modalities=list_tensors_4D[0].shape[0]

	for s in range(nb_scales):
		folder_output_one_scale=folder_output+"/scale"+str(s)

		os.system("mkdir -p " +folder_output_one_scale)

		visualize_patch_3D_one_scale_all_modalities(list_tensors_4D[s],folder_output_one_scale)
		


def visualize_labels_multiscale_patch_3D(list_tensors_3D,folder_output):
	#list_tensors_4D= [tensor4D_image_scale1, tensor4D_image_scale2, tensor4D_image_scale3]

	nb_scales=len(list_tensors_3D)


	for s in range(nb_scales):
		folder_output_one_scale=folder_output+"/scale"+str(s)

		os.system("mkdir -p " +folder_output_one_scale)

		visualize_patch_3D_one_scale_gt(list_tensors_3D[s],folder_output_one_scale)
		



def visualize_patch_3D_one_scale_all_modalities(tensor_4D,folder_output):
	#indexes: modality, z, y, x
	nb_modalities=tensor_4D.shape[0]
	dim_z=tensor_4D.shape[1]


	#print("\n\n\ncontenu du patch une modalite:")

	for num_mod in range(nb_modalities):
		folder_output_one_modality=folder_output+"/modality"+str(num_mod)
		os.system("mkdir -p " +folder_output_one_modality)


		print("contenu du patch une modalite:")
		print(tensor_4D[num_mod,10,10:20,10:20])


		for z in range(dim_z):
			path_result_one_z=folder_output_one_modality+"/slice"+str(z)+".png"
			
			#visualize_slice(tensor_4D[num_mod,z,:,:], path_result_one_z)
			visualize_matrix_gray(tensor_4D[num_mod,z,:,:], path_result_one_z)

			






def visualize_patch_3D_one_scale_gt(tensor_3D,folder_output):
	#indexes: modality, z, y, x

	dim_z=tensor_3D.shape[0]

	
	folder_output_gt=folder_output+"/GT"
	os.system("mkdir -p " +folder_output_gt)

	for z in range(dim_z):
		path_result_one_z=folder_output_gt+"/slice"+str(z)+".png"
		visualize_segmentation_4classes(tensor_3D[z,:,:], path_result_one_z)










def visualize_slice(input_np_2d_float, filename_result):
	height=input_np_2d_float.shape[0]
 	width=input_np_2d_float.shape[1]


	#we want float values between 0 and 1 and in RGBA format

	max_value=windowing_max
	min_value=0.0


	input_correct_ranges=(np.minimum(input_np_2d_float/max_value,1.0)*255.0)

	input_rgb=Image.new("RGBA",(width,height),(0,0,0))

	for y in range(height):
		for x in range(width):
			value=int(input_correct_ranges[y,x])

			#input_rgba.putpixel((x,y),(value,value,value,255))
			input_rgb.putpixel((x,y),(value,value,value))



	input_rgb.save(filename_result,format='png')




def visualize_segmentation_contour(input_np_2d_float, segmentation_2D_np_int, color, filename_result):
	height=input_np_2d_float.shape[0]
 	width=input_np_2d_float.shape[1]



	#img_result=Image.new("RGBA",(width,height),(0,0,0,255))
	

	#we want float values between 0 and 1 and in RGBA format

	max_value=windowing_max
	min_value=0.0


	input_correct_ranges=(np.minimum(input_np_2d_float/max_value,1.0)*255.0)

	
	input_rgb=Image.new("RGBA",(width,height),(0,0,0))

	for y in range(height):
		for x in range(width):
			value=int(input_correct_ranges[y,x])

			#input_rgba.putpixel((x,y),(value,value,value,255))
			input_rgb.putpixel((x,y),(value,value,value))


	#segm_rgba=Image.new("RGBA",(width,height),(0,0,0,255))
	segm_rgb=Image.new("RGBA",(width,height),(0,0,0))
	"""
	for y in range(height):
		for x in range(width):
			segm_rgba.putpixel((x,y),input_rgba.getpixel((x,y)))

	###a checker si le max est bien applique
	"""

	#size_patch=3 <--> patch 3x3
	size_patch=3
	for y in range(height):
		for x in range(width):
			#segm_rgba[:,:,0]=(segmentation_2D_np_int/np.max(segmentation_2D_np_int))*255
			#print(value)

			value_gt=segmentation_2D_np_int[y,x]
			if(value_gt>0):
				#check if we are on the contour
				on_contour=False
				for y_search in range(max(0,y-size_patch/2),min(height-1,y+size_patch/2)+1):
					for x_search in range(max(0,x-size_patch/2),min(width-1,x+size_patch/2)+1):
						if(segmentation_2D_np_int[y_search,x_search]==0):
							#found a neighbor outside the tumor
							on_contour=True
							break

				if(on_contour):
					#print("\n\n\n\n\nCONTOUR")

					if(color=="red"):
						input_rgb.putpixel((x,y),(255,0,0))
					if(color=="green"):
						input_rgb.putpixel((x,y),(0,255,0))
					if(color=="blue"):
						input_rgb.putpixel((x,y),(0,0,255))



	#input_rgba.paste(segm_rgba,(0,0),segm_rgba)
	
	#print("j'enregistre l'image dans "+filename_result)
	#img_result=Image.composite(input_rgb,segm_rgb,segm_rgba)
	
	input_rgb.save(filename_result,format='png')
	#input_rgba.save(filename_result,format='png')





def visualize_3D_segmentation_color_gt_contour(path_folder_flair_axial_patient, tensor3D_output_labels,tensor3D_gt, folder_result, slice_start, color_output, color_gt):
	print("visualizying 3D segmentation...")
	recursive_mkdir(folder_result)
	nb_slices=tensor3D_output_labels.shape[0]
	for s in range(nb_slices):
		num_slice=s+slice_start
		filename_slice_npy=path_folder_flair_axial_patient+"/slice"+str(num_slice)+".npy"
		slice_npy=np.load(filename_slice_npy)
		slice_output_segm=tensor3D_output_labels[s,:,:]
		slice_gt_segm=tensor3D_gt[s,:,:]
		filename_slice_result=folder_result+"/slice"+str(num_slice)+".png"
		visualize_segmentation_color_gt_contour(slice_npy,slice_output_segm,slice_gt_segm, color_output, color_gt,filename_slice_result)
		#visualize_segmentation_contour(slice_npy,slice_segm,filename_slice_result)
	print("done")


#def visualize_segmentation_contour(slice_image_float_npy,slice_segm_int_npy,filename_slice_result):


def visualize_segmentation_color_gt_contour(input_np_2d_float, output_segmentation_2D_np_int,gt_segmentation_2D_np_int, color_output, color_gt, filename_result):
	height=input_np_2d_float.shape[0]
 	width=input_np_2d_float.shape[1]

	max_value=windowing_max
	min_value=0.0


	input_correct_ranges=(np.minimum(input_np_2d_float/max_value,1.0)*255.0)

	
	input_rgb=Image.new("RGBA",(width,height),(0,0,0))

	for y in range(height):
		for x in range(width):
			value=int(input_correct_ranges[y,x])

			#input_rgba.putpixel((x,y),(value,value,value,255))
			input_rgb.putpixel((x,y),(value,value,value))


	segm_rgba=Image.new("RGBA",(width,height),(0,0,0,255))
	segm_rgb=Image.new("RGBA",(width,height),(0,0,0))

	for y in range(height):
		for x in range(width):
			value=int((output_segmentation_2D_np_int[y,x]/np.max(output_segmentation_2D_np_int))*255)
			if(value>0):
				if(color_output=="red"):
					segm_rgba.putpixel((x,y),(255,0,0,200))
					segm_rgb.putpixel((x,y),(255,0,0))
				if(color_output=="green"):
					segm_rgba.putpixel((x,y),(0, 255,0,200))
					segm_rgb.putpixel((x,y),(0,255,0))
				if(color_output=="blue"):
					segm_rgba.putpixel((x,y),(0,0,255,200))
					segm_rgb.putpixel((x,y),(0,0,255))

				if(color_output=="yellow"):
					segm_rgba.putpixel((x,y),(255,255,0,200))
					segm_rgb.putpixel((x,y),(255,255,0))




	img_result=Image.composite(input_rgb,segm_rgb,segm_rgba)
	


	size_patch=3
	for y in range(height):
		for x in range(width):
			#segm_rgba[:,:,0]=(segmentation_2D_np_int/np.max(segmentation_2D_np_int))*255
			#print(value)

			value_gt=gt_segmentation_2D_np_int[y,x]
			if(value_gt>0):
				#check if we are on the contour
				on_contour=False
				for y_search in range(max(0,y-size_patch/2),min(height-1,y+size_patch/2)+1):
					for x_search in range(max(0,x-size_patch/2),min(width-1,x+size_patch/2)+1):
						if(gt_segmentation_2D_np_int[y_search,x_search]==0):
							#found a neighbor outside the tumor
							on_contour=True
							break

				if(on_contour):

					if(color_gt=="red"):
						img_result.putpixel((x,y),(255,0,0))
					if(color_gt=="green"):
						img_result.putpixel((x,y),(0,255,0))
					if(color_gt=="blue"):
						img_result.putpixel((x,y),(0,0,255))



	img_result.save(filename_result,format='png')


def recursive_mkdir(name_folder):
	subfolders=name_folder.split("/")
	constructed_tree=""
	for subfolder in subfolders:
		os.system("mkdir " + constructed_tree+subfolder+"/")
		constructed_tree=constructed_tree+subfolder+"/"



def save_output_layer_txt(output_of_a_layer, filename_result_txt):
	#expected input: 3-dim tensor (indexes: num_channel,y,x)
	file_txt_layer=open(filename_result_txt,"w")
	file_txt_layer.write(" ".join([str(j) for j in output_of_a_layer.shape])+"\n")
	output_flatten=output_of_a_layer.flatten()
	file_txt_layer.write(" ".join([str(j) for j in output_flatten]))
	file_txt_layer.close()



def display_output_layer_all_channels_png(output_of_a_layer,folder_output_all_channels):
	nb_channels=output_of_a_layer.shape[0]
	height=output_of_a_layer.shape[1]
	width=output_of_a_layer.shape[2]
	for num_channel in range(nb_channels):
		fig=plt.figure()
		filename=folder_output_all_channels+"/channel"+str(num_channel)+".png"
		#plt.figure()
		plt.imshow(output_of_a_layer[num_channel,:,:], cmap='gray')
		plt.savefig(filename)
		plt.close(fig)



def show_data(input_tensor,num_image,filename):
	#print("\n\n\nShape of data:")
	#print(net.blobs['data'].data.shape)

	nb_channels_layer=input_tensor.shape[1]
	for c in range(nb_channels_layer):
		#print("\n\n\n\n\nChannel "+str(c)+"\n\n\n")
		#print(net.blobs['data'].data[num_image, c])
		fig=plt.figure()
		plt.imshow(input_tensor[num_image, c,:,:], cmap='gray')
		plt.savefig(filename.replace(".png","") +"_channel"+str(c)+".png")
		plt.close(fig)


def visualize_training_batch_with_gt(input_tensor4D,input_gt_tensor3D,folder_result):
	#display a tensor of 2D multimodal images and their GT segmentations
	os.system("mkdir -p "+folder_result)
	nb_images=input_tensor4D.shape[0]
	nb_channels=input_tensor4D.shape[1]

	for num_image in range(nb_images):
		for c in range(nb_channels):
			visualize_matrix_gray(input_tensor4D[num_image,c,:,:],folder_result+"/data"+str(num_image)+"_channel"+str(c)+".png")
		visualize_segmentation(input_gt_tensor3D[num_image,:,:],folder_result+"/data"+str(num_image)+"_GT.png")







def observed_improvement(list_values_loss,ratio_decrease, nb_elements_estimation):
	#print("i="+str(i))
	nb_elements_list=len(list_values_loss)

	last_index=nb_elements_list-1

	if(nb_elements_list<nb_elements_estimation*2+2):
		print("Not enough elements to estimate the mean loss")
		return True


	recent_mean_loss=mean(list_values_loss,last_index-nb_elements_estimation,last_index)
	previous_mean_loss=mean(list_values_loss,last_index-2*nb_elements_estimation,last_index-nb_elements_estimation-1)

	print("current mean loss="+str(recent_mean_loss))
	print("previous mean loss="+str(previous_mean_loss))

	if(recent_mean_loss<(ratio_decrease*previous_mean_loss)):
		return True

	return False



def mean(list_values, index_start, index_end):
	nb_elements=index_end-index_start+1
	summ=0
	for k in range(index_start,index_end+1):
		#print(k)
		summ=summ+list_values[k]
	result=(float(1)/float(nb_elements))*float(summ)
	return result




def show_stats_composition_training_batch_3D(input_batch_3D_GT, nb_classes):

	list_nb_elements=[]
	nb_total_elements=0

	for c in range(nb_classes):
		nb_el_one_class=(input_batch_3D_GT==c).sum()
		list_nb_elements.append(nb_el_one_class)

		nb_total_elements=nb_total_elements+nb_el_one_class

	for c in range(nb_classes):
		print("Composition of the batch:")
		print("Voxels of class "+str(c)+": "+str(list_nb_elements[c])+" (proportion="+str(float(list_nb_elements[c])/float(nb_total_elements))+")")


def compute_stats_composition_training_batch(input_batch_GT, nb_classes):

	list_nb_elements=[]
	#nb_total_elements=0

	for c in range(nb_classes):
		nb_el_one_class=(input_batch_GT==c).sum()
		list_nb_elements.append(nb_el_one_class)

		#nb_total_elements=nb_total_elements+nb_el_one_class

	#for c in range(nb_classes):
	#	print("Composition of the batch:")
	#	print("Voxels of class "+str(c)+": "+str(list_nb_elements[c])+" (proportion="+str(float(list_nb_elements[c])/float(nb_total_elements))+")")

	return list_nb_elements







def read_tensor_from_hdf5(path_input_hdf5):
	file_input_hdf5=h5py.File(path_input_hdf5,"r")
	result=file_input_hdf5["patient"].value

	return result




def read_time_from_file(filename):
	file_input=open(filename,"r")
	content=file_input.read()
	date=datetime.datetime.strptime(content, "%Y-%m-%d %H:%M:%S.%f")
	file_input.close()
	return date


def read_time_from_str(str_time):
	date=datetime.datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S.%f")
	return date


def write_current_time_to_file(filename):
	file_result=open(filename,"w")

	time_now=datetime.datetime.now()

	file_result.write(str(time_now))

	file_result.close()


def write_list_tensors_npy_to_file(list_tensors,folder_result):
	os.system("mkdir -p "+folder_result)

	for k in range(len(list_tensors)):
		tensor=list_tensors[k]
		filename_result_tensor=folder_result+"/tensor"+str(k)+".npy"
		np.save(filename_result_tensor,tensor)



def load_list_tensors(folder_tensors):

	list_tensors=[]

	filenames=os.listdir(folder_tensors)

	nb_tensors=len(filenames)

	for k in range(nb_tensors):
		filename_tensor=folder_tensors+"/tensor"+str(k)+".npy"
		list_tensors.append(np.load(filename_tensor))


	return list_tensors



def write_list_to_file(list_numbers,filename_result):
	list_str=[str(list_numbers[k]) for k in range(len(list_numbers))]

	file_result=open(filename_result,"w")
	file_result.write(" ".join(list_str))
	file_result.close()


def load_list_float(filename):
	result=[]

	file_input=open(filename,"r")
	content_str=file_input.read().split(" ")
	file_input.close()

	



	result=[float(content_str[k]) for k in range(len(content_str))]
	return result





def list_int_from_str(listt):
	return [int(listt[k]) for k in range(len(listt))]


def list_without_repetitions(listt):
	return list(set(listt))

	

def find_names_features(list_keywords_features,folder_hdf5_all_modalities_and_features):
	filenames=os.listdir(folder_hdf5_all_modalities_and_features)

	result=[]

	for f in filenames:
		for keyword in list_keywords_features:
			if(keyword in f):
				result.append(f)

	result.sort()
	return result




def save_estimates_mean_and_std_net(net, folder_output_parameters):
	print("saving mean and std in the folder "+folder_output_parameters)
	for layer in net.list_layers:
		if(layer.type=="conv" and layer.apply_bn):
			layer.save_mean_and_std(folder_output_parameters)











def visualize_segmentation_color(input_np_2d_float, segmentation_2D_np_int, color, filename_result):
	height=input_np_2d_float.shape[0]
 	width=input_np_2d_float.shape[1]



	#img_result=Image.new("RGBA",(width,height),(0,0,0,255))
	

	#we want float values between 0 and 1 and in RGBA format

	max_value=windowing_max
	min_value=0.0


	input_correct_ranges=(np.minimum(input_np_2d_float/max_value,1.0)*255.0)

	
	input_rgb=Image.new("RGBA",(width,height),(0,0,0))

	for y in range(height):
		for x in range(width):
			value=int(input_correct_ranges[y,x])

			#input_rgba.putpixel((x,y),(value,value,value,255))
			input_rgb.putpixel((x,y),(value,value,value))


	segm_rgba=Image.new("RGBA",(width,height),(0,0,0,255))
	segm_rgb=Image.new("RGBA",(width,height),(0,0,0))
	"""
	for y in range(height):
		for x in range(width):
			segm_rgba.putpixel((x,y),input_rgba.getpixel((x,y)))

	###a checker si le max est bien applique
	"""

	for y in range(height):
		for x in range(width):
			#segm_rgba[:,:,0]=(segmentation_2D_np_int/np.max(segmentation_2D_np_int))*255
			#print(value)

			value=int((segmentation_2D_np_int[y,x]/np.max(segmentation_2D_np_int))*255)
			if(value>0):
				if(color=="red"):
					segm_rgba.putpixel((x,y),(value,0,0,200))
					segm_rgb.putpixel((x,y),(value,0,0))
				if(color=="green"):
					segm_rgba.putpixel((x,y),(0, value,0,200))
					segm_rgb.putpixel((x,y),(0,value,0))
				if(color=="blue"):
					segm_rgba.putpixel((x,y),(0,0,value,200))
					segm_rgb.putpixel((x,y),(0,0,value))



	#input_rgba.paste(segm_rgba,(0,0),segm_rgba)
	
	#print("j'enregistre l'image dans "+filename_result)
	img_result=Image.composite(input_rgb,segm_rgb,segm_rgba)
	
	img_result.save(filename_result,format='png')
	#input_rgba.save(filename_result,format='png')

	