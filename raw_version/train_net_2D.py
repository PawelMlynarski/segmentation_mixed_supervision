
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
import theano
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import bilinear_upsampling


folder_all_trainings="trainings_mixed_supevision"




folder_results_all_tests="results_joint"



#path_results_all_tests="tests_protontherapy_fevrier2019"
paths_folder_files_launch_tests="lists_tests/tests2019"


launch_test=True




joint_or_standard=sys.argv[1]

if((joint_or_standard!="joint") and (joint_or_standard!="standard")):
	print("Argument 1: precise 'joint' or 'standard'")
	sys.exit(1)

nb_classes=2





name_net=sys.argv[4]












#type_slice="axial"
#type_slice="sagittal"
#type_slice="coronal"




if(joint_or_standard!="standard"):
	weight_segmentation=float(sys.argv[6])
	print("\n\n\n\n\n\nweight_segmentation: "+str(weight_segmentation))







str_region_segmentation=sys.argv[7]

if((str_region_segmentation!="whole_tumor") and (str_region_segmentation!="tumor_core") and (str_region_segmentation!="enhancing_core")):
	print("unknown region:"+str_region_segmentation)
	sys.exit(1)



#list_modalities=["T2_FLAIR","T1"]
list_modalities=["T2_FLAIR","T1", "T1c","T2"]
#list_modalities=["T1c"]



read_learning_rate=True 

pad_zeros_input=False
#pad_zeros_input=True


noise_freq=2
noise_std_deviation=0




#margin_boundary_classes=0








pad_zeros_input=True
target_shape_input_image=[300,300]




type_slice=sys.argv[5]




weights_classes=[0.7,0.3]



if((joint_or_standard!="joint") and (joint_or_standard!="standard")):
	print("Specify 'joint', 'standard', 'fine_tuning' or 'classification'")
	sys.exit(1)





reference_test=sys.argv[2]
num_fold=int(sys.argv[3])


#reference_test=1
#num_fold=5



filename_training_with_GT="lists_images/test"+str(reference_test)+"/fold"+str(num_fold)+"/training_patients_flair_"+type_slice+"_with_gt.txt"
filename_training_without_GT="lists_images/test"+str(reference_test)+"/fold"+str(num_fold)+"/training_patients_flair_"+type_slice+"_without_gt.txt"






momentum=0.9






learning_rate=0.25
#learning_rate=0.125







#use_bn_in_fc_layers=True
use_bn_in_fc_layers=False

#take_several_batches=False
take_several_batches=True



#use_axial_flips=False
use_axial_flips=True





freq_check_learning_rate=500
ratio_decrease_loss=0.99
#number of 
nb_losses_estimatation=245
factor_decrease_lr=0.5










print("\n\n\n\n\nTYPE SLICE:"+type_slice)
print("\n\n\n")





		
apply_bn_conv_layers=True


		
nb_batches_one_step=10

freq_switch_batch=1




if(joint_or_standard!="standard"):
	batch_size=10

	nb_images_label1_with_gt=4
	nb_images_label0=2
	nb_images_label1_without_gt=4


else:


	batch_size=6

	nb_images_label1_with_gt=4
	nb_images_label0=2
	nb_images_label1_without_gt=0












print("\n\n\n\n\n")
print("nb_batches_one_step="+str(nb_batches_one_step))
print("nb_images_label1_with_gt="+str(nb_images_label1_with_gt))
print("nb_images_label0="+str(nb_images_label0))
print("nb_images_label1_without_gt="+str(nb_images_label1_without_gt))
print("\n\n\n\n\n")
#weight_decay=0.004
weight_decay=0.0


name_standard="standard_"

if((joint_or_standard=="fine_tuning")):
	name_test="fine_tuning_"

if((joint_or_standard=="joint")):
	name_test="joint_"


if((joint_or_standard=="classification")):
	name_test="classification_"


if((joint_or_standard=="standard")):
	name_test="standard_"

	#for fine-tuning
	


name_test=name_test+"_"+name_net+ "_test"+str(reference_test)+"_fold"+str(num_fold)

name_standard=name_standard+"_"+name_net+ "_test"+str(reference_test)+"_fold"+str(num_fold)


if(weights_classes!=None):
	name_test=name_test+"_weighted"+string.join([str(k) for k in weights_classes],"_")

	name_standard=name_standard+"_weighted"+string.join([str(k) for k in weights_classes],"_")





if((joint_or_standard=="joint") or (joint_or_standard=="fine_tuning")):
	name_test=name_test+"_weight_segm"+str(weight_segmentation)

name_test=name_test+"_"+type_slice
name_standard=name_standard+"_"+type_slice






if(pad_zeros_input):
	name_test=name_test+"_padded"
	name_standard=name_standard+"_padded"











if(use_axial_flips):
	name_test=name_test+"_axial_flips"
	name_standard=name_standard+"_axial_flips"

if(len(list_modalities)<4):
	name_test=name_test+"_"+"_".join(list_modalities)
	name_standard=name_standard+"_"+"_".join(list_modalities)






if(take_several_batches):
	name_test=name_test+"_several_batches"+str(nb_batches_one_step)
	name_standard=name_standard+"_several_batches"+str(nb_batches_one_step)




name_test=name_test+"_"+str_region_segmentation
name_standard=name_standard+"_"+str_region_segmentation


multiplication_input=1.0


if(not(apply_bn_conv_layers)):
	name_test=name_test+"_no_batch_norm"
	name_standard=name_standard+"_no_batch_norm"


if(multiplication_input!=1.0):
	name_test=name_test+"_multiplication"+str(multiplication_input)
	name_standard=name_standard+"_multiplication"+str(multiplication_input)


"""
if(learning_rate!=0.01):
	name_test=name_test+"_lr"+str(learning_rate)

"""


if(batch_size!=(nb_images_label1_with_gt+nb_images_label0+nb_images_label1_without_gt)):
	print("PROBLEM WITH THE BATCH SIZE")
	sys.exit(2)






folder_all_networks="networks/joint2018"








folder_results=folder_all_trainings+"/"+name_test
os.system("mkdir -p "+folder_results)


if(sys.argv[-2]=="communication"):
	#create the file with the path for the folder_results and quit

	filename_communication_start=sys.argv[-1]

	file_communication_start=open(filename_communication_start,"w")

	file_communication_start.write(folder_results)

	file_communication_start.close()
	print("ok moi j'ai cree le fichier")
	sys.exit(0)



num_parameters=find_current_parameters(folder_results)
path_parameters=folder_results+"/parameters"+str(num_parameters)
iteration_start=num_parameters






"""
#standard for axial slices
slice_start_axial=45
slice_end_axial=125
"""




#ignore empty slices of BRATS	

slice_start_axial=15
slice_end_axial=130


slice_start_sagittal=55
slice_end_sagittal=180


slice_start_coronal=60
slice_end_coronal=200



if(type_slice=="axial"):
	slice_start=slice_start_axial
	slice_end=slice_end_axial


elif(type_slice=="sagittal"):
	slice_start=slice_start_sagittal
	slice_end=slice_end_sagittal

elif(type_slice=="coronal"):
	slice_start=slice_start_coronal
	slice_end=slice_end_coronal
else:
	print("unknown type of slice")
	sys.exit(2)







#construct the lists of paths of folders (2 lists: path_folder_patient + path_file_labels)

list_folders_patients_training_with_gt,list_files_labels_each_training_patient_with_gt=create_lists_folders_slices_and_files_labels_of_slices(filename_training_with_GT)





if(joint_or_standard!="standard"):
	list_folders_patients_training_without_gt,list_files_labels_each_training_patient_without_gt=create_lists_folders_slices_and_files_labels_of_slices(filename_training_without_GT)

else:
	list_folders_patients_training_without_gt=[]
	list_files_labels_each_training_patient_without_gt=[]

#here we read the first modality
first_image=np.load(list_folders_patients_training_with_gt[0]+"/slice0.npy")

batch_shape=[batch_size,len(list_modalities),first_image.shape[0],first_image.shape[1]]




























lists_images_training=create_lists_images_training(list_folders_patients_training_with_gt,list_files_labels_each_training_patient_with_gt,list_folders_patients_training_without_gt,list_files_labels_each_training_patient_without_gt, slice_start, slice_end , nb_classes,str_region_segmentation=str_region_segmentation)
#reminder: 3 lists: images labeled 1 with gt, images labeled 0, images labeled 1 without gt
#each entry: [path_flair,path_gt,list_present_classes] with list_present_classes in the standard form ([0,2,3])









list_images_batch=choose_images_batch(lists_images_training,[0,0,0], nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt)











if(read_learning_rate):
	#read the learning rate
	lr=find_recent_learning_rate(folder_results)

	if(lr!=(-1.0)):
		learning_rate=lr
else:
	learning_rate=new_learning_rate
	print("\n\n\n\nNo reading of the learning rate, fixed at "+str(new_learning_rate))






nb_images_gt= nb_images_label1_with_gt + nb_images_label0



if((joint_or_standard=="standard") and (nb_images_label1_without_gt>0)):
	print("Images without GT provided for the standard training")
	sys.exit(4)






if((joint_or_standard=="joint") and (nb_images_label1_without_gt==0)):
	print("Images without GT no provided for the joint training")
	sys.exit(4)




#ATTENTION
use_joint_training=(joint_or_standard!="standard")



if(use_joint_training):
	print("\n\n\n\n\nJOINT TRAINING")


net=NeuralNet2D(folder_all_networks+"/"+name_net+".prototxt",str_name=name_net,nb_classes=nb_classes, weight_segmentation=weight_segmentation, joint_training=use_joint_training, nb_images_gt=nb_images_gt, input_tensor_shape=batch_shape, momentum=momentum,learning_rate=learning_rate, weight_decay=weight_decay,weights_classes=weights_classes,apply_bn_conv_layers=apply_bn_conv_layers,pad_zeros_input=pad_zeros_input,target_shape_input_image=target_shape_input_image)



nn_offset_y=net.total_offset_y
nn_offset_x=net.total_offset_x

net.print_layers()










print("definition des fonctions")


if((joint_or_standard=="joint") or (joint_or_standard=="fine_tuning")):
	compute_gradients_loss_wrt_parameters=theano.function([net.input_network_0, net.tensor_labels_0]+net.list_vectors_labels, outputs=net.gradients_loss_wrt_parameters+[net.loss])
if(joint_or_standard=="standard"):
	compute_gradients_loss_wrt_parameters=theano.function([net.input_network_0, net.tensor_labels_0],outputs=net.gradients_loss_wrt_parameters+[net.loss])

i_theano=T.iscalar('i_theano')


if((joint_or_standard=="joint") or (joint_or_standard=="fine_tuning")):
	compute_all_scores_and_losses=theano.function([net.input_network_0, net.tensor_labels_0]+net.list_vectors_labels,outputs=[net.output_segmentation_scores, net.loss,net.segmentation_loss,net.classification_loss, net.accuracy])
if(joint_or_standard=="standard"):
	compute_all_scores_and_losses=theano.function([net.input_network_0, net.tensor_labels_0],outputs=[net.output_segmentation_scores,net.loss])



print("\n\n\n\nfonctions definies")




if(joint_or_standard=="fine_tuning"):
	joint_or_standard="joint"








if(num_parameters>0):
	print("\n\n\n\n\n\nje lis  les parametres \n\n\n\n\n")
	net.set_parameters_all_layers(path_parameters)
	print("\n\n\n\n\n\nj'ai bien lu les parametres \n\n\n\n\n")











recursive_mkdir(folder_results)
#write th optimization parameters to a file
num_opt_parameters=0
for k in range(50):
	num_opt_parameters=k
	filename_opt_parameters=folder_results+"/opt_parameters"+str(num_opt_parameters)+".txt"
	if(not(os.path.isfile(filename_opt_parameters))):
		break

file_opt_parameters=open(filename_opt_parameters,"w")
file_opt_parameters.write("Learning rate: "+str(learning_rate)+"\n")
file_opt_parameters.write("Momentum: "+str(momentum)+"\n")
file_opt_parameters.write("Weight decay: "+str(weight_decay)+"\n")
if(weights_classes!=None):
	file_opt_parameters.write("Weights classes: "+string.join([str(k) for k in weights_classes]," ")+"\n")
file_opt_parameters.write("path_parameters start: "+str(path_parameters)+"\n")
file_opt_parameters.write("Slices: ("+str(slice_start)+ ","+str(slice_end)+ ")\n")
file_opt_parameters.write("Modalitites: "+string.join([str(k) for k in list_modalities]," ")+"\n")
file_opt_parameters.write("Batch size: "+str(batch_size)+"\n")
file_opt_parameters.write("Freq switch batch: "+str(freq_switch_batch)+"\n")
file_opt_parameters.write("nb_images_label1_with_gt: "+str(nb_images_label1_with_gt)+"\n")
file_opt_parameters.write("nb_images_label0: "+str(nb_images_label0)+"\n")
file_opt_parameters.write("nb_images_label1_without_gt: "+str(nb_images_label1_without_gt)+"\n")
file_opt_parameters.write("noise_freq: "+str(noise_freq)+"\n")
file_opt_parameters.write("noise_std_deviation: "+str(noise_std_deviation)+"\n")
file_opt_parameters.write("filename_training_with_GT: "+str(filename_training_with_GT)+"\n")
file_opt_parameters.write("slice_start: "+str(slice_start)+"\n")
file_opt_parameters.write("slice_end: "+str(slice_end)+"\n")
#file_opt_parameters.write("margin_boundary_classes: "+str(margin_boundary_classes)+"\n")
if((joint_or_standard=="joint") or (joint_or_standard=="classification")):
	file_opt_parameters.write("filename_training_without_GT: "+str(filename_training_without_GT)+"\n")
file_opt_parameters.close()




#file for the values of learning rates accross the training
filename_values_lr=""
for k in range(50):
	num_opt_parameters=k
	filename_values_lr=folder_results+"/learning_rates"+str(k)+".txt"
	if(not(os.path.isfile(filename_values_lr))):
		break









def show_training_batch(input_batch, folder_results, num_iteration):
	nb_images=input_batch.shape[0]

	recursive_mkdir(folder_results+"/batch"+str(num_iteration))
	for num_image in range(nb_images):
		show_data(input_batch,num_image,folder_results+"/batch"+str(num_iteration)+"/data"+str(num_image)+".png")


def show_training_batch_and_gt(input_batch, tensor_labels, folder_results, num_iteration):
	nb_images=input_batch.shape[0]

	nb_modalities=input_batch.shape[1]
	#recursive_mkdir(folder_results+"/batch"+str(num_iteration))
	#for num_image in range(nb_images):
	#	show_data(input_batch,num_image,folder_results+"/batch"+str(num_iteration)+"/data"+str(num_image)+".png")

	folder_batch=folder_results+"/batch"+str(num_iteration)
	os.system("mkdir -p "+folder_batch)

	for num_image in range(nb_images):
		for num_modality in range(nb_modalities):
			path_one_modality=folder_batch+"/img"+str(num_image)+"_mod"+str(num_modality)+".png"
			image_one_modality=input_batch[num_image,num_modality]
			visualize_matrix_gray(image_one_modality,path_one_modality)



		path_gt=folder_batch+"/img"+str(num_image)+"_gt.png"
		image_gt=tensor_labels[num_image]
		visualize_segmentation_4classes(image_gt,path_gt)



#fonctions for the training


def compute_gradients_parameters_one_batch_joint(input_batch, tensor_labels_batch, list_vectors_all_labels_batch, b_compute_stats):
	loss=0.0
	precision=0.0
	recall=0.0
	segmentation_loss=0.0
	classification_loss=0.0
	accuracy=0.0
	stats_all_classes=[]












	target_shape_outputs_one_batch=[nb_images_gt,first_image.shape[0],first_image.shape[1]]








	
	tmp=compute_gradients_loss_wrt_parameters(input_batch, tensor_labels_batch, list_vectors_all_labels_batch[0])


	


	list_gradients=tmp[:-1]
	loss=tmp[-1]

	if(b_compute_stats):
		
		scores, loss, segmentation_loss, classification_loss, accuracy=compute_all_scores_and_losses(input_batch, tensor_labels_batch, list_vectors_all_labels_batch[0])
	
		output_labels=np.argmax(scores,axis=1)
		

		if(pad_zeros_input):
			
			output_labels=adjust_shape_tensor3(output_labels,target_shape_outputs_one_batch,net.nb_zeros_top, net.nb_zeros_left, nn_offset_y,nn_offset_x)

			recall,precision=compute_recall_and_precision(output_labels, tensor_labels_batch)
			stats_all_classes=compute_stats_all_subclasses_output_tensor_labels(output_labels, tensor_labels_batch,nb_classes)


		else:
			#output_labels=add_zero_padding(output_labels,tensor_labels_batch.shape)
			tensor_labels_batch_cropped=tensor_labels_batch[:,nn_offset_y:(nn_offset_y+output_labels.shape[1]),nn_offset_x:(nn_offset_x+output_labels.shape[2])]
			recall,precision=compute_recall_and_precision(output_labels, tensor_labels_batch_cropped)
			stats_all_classes=compute_stats_all_subclasses_output_tensor_labels(output_labels, tensor_labels_batch_cropped,nb_classes)
	
	#print("LOSS="+str(loss))
	return (list_gradients, loss, segmentation_loss, classification_loss, recall, precision, accuracy,stats_all_classes)




def compute_gradients_parameters_one_batch_standard(input_batch, tensor_labels_batch, b_compute_stats):
	loss=0.0
	current_loss=np.array([0],dtype=theano.config.floatX)
	precision=0.0
	recall=0.0
	stats_all_classes=[]
	
	#list_gradients,loss=compute_gradients_loss_wrt_parameters(input_batch, tensor_labels_batch)
	tmp=compute_gradients_loss_wrt_parameters(input_batch, tensor_labels_batch)

	list_gradients=tmp[:-1]
	loss=tmp[-1]


	target_shape_outputs_one_batch=[batch_size,first_image.shape[0],first_image.shape[1]]


	if(b_compute_stats):
		scores, loss=compute_all_scores_and_losses(input_batch, tensor_labels_batch)
		output_labels=np.argmax(scores,axis=1)
		
		if(pad_zeros_input):
			
			output_labels=adjust_shape_tensor3(output_labels,target_shape_outputs_one_batch,net.nb_zeros_top, net.nb_zeros_left, nn_offset_y,nn_offset_x)

			recall,precision=compute_recall_and_precision(output_labels, tensor_labels_batch)
			stats_all_classes=compute_stats_all_subclasses_output_tensor_labels(output_labels, tensor_labels_batch,nb_classes)


		else:
			#output_labels=add_zero_padding(output_labels,tensor_labels_batch.shape)
			tensor_labels_batch_cropped=tensor_labels_batch[:,nn_offset_y:(nn_offset_y+output_labels.shape[1]),nn_offset_x:(nn_offset_x+output_labels.shape[2])]
			recall,precision=compute_recall_and_precision(output_labels, tensor_labels_batch_cropped)
			stats_all_classes=compute_stats_all_subclasses_output_tensor_labels(output_labels, tensor_labels_batch_cropped,nb_classes)


		
	

	#print("LOSS="+str(loss))
	return (list_gradients, loss, recall, precision,stats_all_classes)



def compute_gradients_parameters_one_batch_classification(input_batch, list_vectors_all_labels_batch,b_compute_stats):
	loss=0.0	
	accuracy=0.0
	




	
	tmp=compute_gradients_loss_wrt_parameters(input_batch, list_vectors_all_labels_batch[0])




	list_gradients=tmp[:-1]
	loss=tmp[-1]

	if(b_compute_stats):
		
		loss, accuracy=compute_all_scores_and_losses(input_batch, list_vectors_all_labels_batch[0])
		
	
	return (list_gradients, loss, accuracy)



	




def create_tensor_input_tensor_labels_vector_all_labels_all_processing(lists_images_training, list_indexes_start, batch_shape, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt, list_modalities):
	




	nb_images_batch=nb_images_label1_with_gt+nb_images_label0+nb_images_label1_without_gt



	#choose the images for the batch

	list_images_batch=choose_images_batch(lists_images_training,list_indexes_start, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt)

	#format of the list: [ [path_flair, path_gt, binary_vector_present_classes]], ...]

	#check if the batch contains all tumor subclasses

	all_classes_present=False


	count_rejected_batches=0

	while(not(all_classes_present)):

		count_rejected_batches=count_rejected_batches+1

		if(count_rejected_batches>30):
			print("Something is wrong with the batches")
			sys.exit(3)

		


		all_classes_present=True


		
		for cl in range(1,nb_classes):
			cl_present=False

		

			for num_image in range(nb_images_batch):
				#if(list_images_batch[num_image][2][index_class]==1):
				if(cl in list_images_batch[num_image][2]):
					cl_present=True
					break


			if(not(cl_present)):
				all_classes_present=False
				print("Class "+str(cl)+" not present, reject the batch")
				#sys.exit(1)
				break






	


	input_batch, tensor_labels_batch, list_vectors_all_labels_batch=create_batch_given_list_of_images_batch(list_images_batch, batch_shape, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt, nb_classes=nb_classes, list_modalities=list_modalities)




	#parse labels


	tensor_labels_batch=regroup_labels_one_tensor(input_tensor=tensor_labels_batch,str_region_segmentation=str_region_segmentation)



	#axial flips
	if(use_axial_flips):
		if(type_slice=="axial"):
			input_batch, tensor_labels_batch=generate_axial_flips_batch_and_gt(input_batch, tensor_labels_batch, proportion_flip=0.3)
		elif(type_slice=="coronal"):
			
			input_batch, tensor_labels_batch=generate_axial_flips_batch_and_gt(input_batch, tensor_labels_batch, proportion_flip=0.3)		
		else:
			if(type_slice!="sagittal"):
				print("Unknown type of slice")
				sys.exit(2)
			

	if(noise_std_deviation!=0):
		noise=np.asarray(np.random.normal(0,noise_std_deviation,size=input_batch.shape),dtype=input_batch.dtype)
		input_batch=input_batch+noise

	if(multiplication_input!=1.0):
		input_batch=input_batch*multiplication_input

	

	return (input_batch, tensor_labels_batch, list_vectors_all_labels_batch)


def create_list_of_batches_and_labels(nb_batches, lists_images_training, list_indexes_start, batch_shape, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt, list_modalities):
	
	result=[]
	for b in range(nb_batches):

	
		input_batch, tensor_labels_batch, list_vectors_all_labels_batch=create_tensor_input_tensor_labels_vector_all_labels_all_processing(lists_images_training, list_indexes_start, batch_shape, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt, list_modalities)

		result.append([input_batch, tensor_labels_batch, list_vectors_all_labels_batch])


	return result








def compute_gradients_parameters_several_batches2(nb_batches_one_iteration,b_compute_stats, batch_shape, lists_images_training_input=None, list_indexes_start_input=None,  list_batches=None):
	#version online: iteratively create the batches, compute the gradients and losses if requested
	#version offline: the same but when the batches are already created
	

	precision=0.0
	recall=0.0
	segmentation_loss=0.0
	classification_loss=0.0


	segmentation_loss_current=0.0
	classification_loss_current=0.0
	
	accuracy_current=0.0
	precision_current=0.0
	recall_current=0.0
	stats_all_classes_current=[]

	accuracy=0.0
	loss=0.0

	list_gradients=[]

	
	for b in range(nb_batches_one_iteration):

		if(list_batches==None):
			print("not implemented")
			sys.exit(1)
	
		else:
			el=list_batches[b]
			input_batch=el[0]
			tensor_labels_batch=el[1]
			list_vectors_all_labels_batch=el[2]	


		

		if(joint_or_standard=="joint"):
			gradients_parameters_current,loss_current, segmentation_loss_current, classification_loss_current, recall_current, precision_current, accuracy_current ,stats_all_classes_current=compute_gradients_parameters_one_batch_joint(input_batch, tensor_labels_batch, list_vectors_all_labels_batch, b_compute_stats)
		if(joint_or_standard=="standard"):
			gradients_parameters_current,loss_current, recall_current, precision_current,stats_all_classes_current=compute_gradients_parameters_one_batch_standard(input_batch, tensor_labels_batch, b_compute_stats)
		



		loss=loss+loss_current
		precision=precision+precision_current
		recall=recall+recall_current

		if(joint_or_standard=="joint"):
			segmentation_loss=segmentation_loss+segmentation_loss_current
			classification_loss=classification_loss+classification_loss_current

		
		accuracy=accuracy+accuracy_current

		
		if(b==0):
			stats_all_classes=list(stats_all_classes_current)
			list_gradients=list(gradients_parameters_current)

			#convert the type
			for j in range(len(list_gradients)):
				list_gradients[j]=np.asarray(list_gradients[j],dtype=theano.config.floatX)

		else:

			#sum of gradients (not mean)
			for j in range(len(list_gradients)):
				##Old: INEFFICACE (correction done)
				list_gradients[j]=list_gradients[j]+np.asarray(gradients_parameters_current[j],dtype=theano.config.floatX)


			for i in range(len(stats_all_classes)):
				stats_all_classes[i][0]=stats_all_classes[i][0]+stats_all_classes_current[i][0]
				stats_all_classes[i][1]=stats_all_classes[i][1]+stats_all_classes_current[i][1]
				stats_all_classes[i][2]=stats_all_classes[i][2]+stats_all_classes_current[i][2]


	#loss=loss/float(nb_batches_one_iteration)
	recall=recall/float(nb_batches_one_iteration)
	precision=precision/float(nb_batches_one_iteration)
	accuracy=accuracy/float(nb_batches_one_iteration)

	for i in range(len(stats_all_classes)):
		stats_all_classes[i][0]=stats_all_classes[i][0]/float(nb_batches_one_iteration)
		stats_all_classes[i][1]=stats_all_classes[i][1]/float(nb_batches_one_iteration)
		stats_all_classes[i][2]=stats_all_classes[i][2]/float(nb_batches_one_iteration)



	#normalize
	norm_gradient=compute_norm_gradient(list_gradients)
	#print("norm gradient:" +str(norm_gradient))

	for j in range(len(list_gradients)):
		list_gradients[j]=list_gradients[j]/norm_gradient



	if(joint_or_standard=="joint"):
		return (list_gradients, loss, segmentation_loss, classification_loss, recall, precision, accuracy,stats_all_classes)

		
	elif(joint_or_standard=="standard"):
		return (list_gradients, loss, recall, precision,stats_all_classes)
		

	else:
		print("ERROR: unknown type of the network")
		sys.exit(5)




	



def update_parameters(net, gradients_parameters,previous_v):

	v=[]
	k=-1

	learning_rate=np.asarray([net.learning_rate],dtype=theano.config.floatX)
	momentum=np.asarray([net.momentum],dtype=theano.config.floatX)

	for parameter in net.parameters:
		k=k+1
		current_value=parameter.get_value()

		if(previous_v!=[]):
			v.append(previous_v[k]*momentum-learning_rate*gradients_parameters[k])
			#v.append(previous_v[k]*momentum-(1.0-momentum)*learning_rate*gradients_parameters[k])
			#print("ici")
		else:
			v.append(-learning_rate*gradients_parameters[k])

		parameter.set_value(current_value+v[k])



	return v



def train_network(net, lists_images_training, nb_iterations, batch_size, nb_batches_one_step, nb_images_gt_each_batch, freq_display, freq_compute_stats, folder_results,list_indexes_start=None):
	recursive_mkdir(folder_results+"/loss")
	recursive_mkdir(folder_results+"/recall")
	recursive_mkdir(folder_results+"/precision")
	recursive_mkdir(folder_results+"/accuracy")
	recursive_mkdir(folder_results+"/loss_segmentation")
	recursive_mkdir(folder_results+"/loss_classification")




	precision_subclasses_training=[]
	recall_subclasses_training=[]
	dice_subclasses_training=[]


	for cl in range(nb_classes):
		recursive_mkdir(folder_results+"/recall_class"+str(cl))
		recursive_mkdir(folder_results+"/precision_class"+str(cl))
		recursive_mkdir(folder_results+"/dice_class"+str(cl))
		precision_subclasses_training.append([])
		recall_subclasses_training.append([])
		dice_subclasses_training.append([])



	v=[]



	



	loss_training=[]



	if(num_parameters>0):
		filename_loss=path_parameters+"/loss.txt"
		loss_training=load_list_float(filename_loss)
		folder_v=path_parameters+"/v"
		v=load_list_tensors(folder_v)






	loss_classification_training=[]
	loss_segmentation_training=[]
	recall_training=[]
	precision_training=[]
	accuracy_training=[]

	loss_test=[]
	recursive_mkdir(folder_results)

	b_compute_stats=True
	

	freq_save_parameters=400



	if(nb_batches_one_step>20):
		freq_save_parameters=25



	freq_check_learning_rate_local=freq_check_learning_rate
	ratio_decrease_loss_local=ratio_decrease_loss
	nb_losses_estimatation_local=nb_losses_estimatation

	
	count_switch_lr=0

	filename_communication=folder_results+"/communication.txt"

	write_current_time_to_file(filename_communication)


	#freq_signal=5
	freq_signal=3

	
	
	if(list_indexes_start==None):
		list_indexes_start=[0, 0, 0]




	#this list will contain couples (iteration, learning_rate)
	list_learning_rates=[(iteration_start, net.learning_rate)]







	k=0
	#do the optimization
	#for j in range(nb_iterations):
	for i in range(iteration_start, nb_iterations):
		#i=j+iteration_start
		j=i-iteration_start
		if(i%20==0):
			print("\n\nITERATION "+str(i))

		if(i%freq_compute_stats==0):
			b_compute_stats=True
		else:
			b_compute_stats=False



		if(j%freq_signal==0):
			write_current_time_to_file(filename_communication)




		#check the learning rate and update it if needed


		if(j%freq_check_learning_rate_local==0):
			loss_improved=observed_improvement(loss_training,ratio_decrease_loss, nb_losses_estimatation)


			if(not(loss_improved)):
				if(count_switch_lr==0):
					#decrease the learning rate and display it
					#net.learning_rate=max(net.learning_rate*factor_decrease_lr,0.005)
					net.learning_rate=max(net.learning_rate*factor_decrease_lr,0.001)
					print("Iteration "+str(i)+", new learning rate:"+str(net.learning_rate))
					count_switch_lr=count_switch_lr+1

					list_learning_rates.append((i,net.learning_rate))

					file_lr=open(filename_values_lr,"w")
					file_lr.write(str(list_learning_rates))
					file_lr.close()

				else:
					freq_check_learning_rate_local=freq_check_learning_rate_local+100
					nb_losses_estimatation_local=freq_check_learning_rate_local/2 - 5
					ratio_decrease_loss_local=min(ratio_decrease_loss_local+0.001, 0.995)

					print("Iteration "+str(i)+", the new learning rate was not enough to decrease the loss: now I will check every "+str(freq_check_learning_rate_local)+ " iterations and I want a decrease of "+str(1.0- ratio_decrease_loss) )
					count_switch_lr=0

				

			else:
				#remember that the last time the loss improved
				count_switch_lr=0


	



		#remark: list_indexes_start will be changed by other functions


		if(take_several_batches):
			#version offline: fast but consumes the memory

			if(j%freq_switch_batch==0):
				list_batches=create_list_of_batches_and_labels(nb_batches_one_step, lists_images_training, list_indexes_start, batch_shape, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt, list_modalities)
			
			if(joint_or_standard=="joint"):
				gradients_parameters,loss, loss_segmentation, loss_classification, recall, precision, accuracy,stats_all_classes=compute_gradients_parameters_several_batches2(nb_batches_one_step,b_compute_stats, batch_shape, lists_images_training_input=None, list_indexes_start_input=None,list_batches=list_batches)
			
			if(joint_or_standard=="standard"):
				gradients_parameters,loss, recall, precision,stats_all_classes=compute_gradients_parameters_several_batches2(nb_batches_one_step,b_compute_stats, batch_shape, lists_images_training_input=None, list_indexes_start_input=None,list_batches=list_batches)
			


			if(loss!=loss):

				print("Loss became NaN")
				sys.exit(1)

		else:

			print("not implemented")
			sys.exit(1)



		#update the parameters given the computed gradients of the loss wrt to parameters
		v=update_parameters(net, gradients_parameters,v)
		
		if(i>1):
			loss_training.append(float(loss))
			if(b_compute_stats):

				#loss_training.append(float(loss))

				if((joint_or_standard=="joint")):
					loss_segmentation_training.append(float(loss_segmentation))
					loss_classification_training.append(float(loss_classification))
					accuracy_training.append(float(accuracy))


				if((joint_or_standard=="joint") or (joint_or_standard=="standard")):
					recall_training.append(float(recall))
					precision_training.append(float(precision))

					for cl in range(nb_classes):
						precision_subclass=stats_all_classes[cl][0]
						recall_subclass=stats_all_classes[cl][1]
						dice_subclass=stats_all_classes[cl][2]

						recall_subclasses_training[cl].append(float(recall_subclass))
						precision_subclasses_training[cl].append(float(precision_subclass))
						dice_subclasses_training[cl].append(float(dice_subclass))

					



		
				

		#input_net_real, labels_real=create_tensor(list_paths_and_labels_test, 0, 165)
		#loss_test.append(compute_loss(input_net_real,labels_real))
		if((i%freq_compute_stats==0)):
			print("LOSS="+str(loss))


		if((i>1) and (i%freq_display==0)):
			show_evolution(loss_training,folder_results+"/loss/loss_training"+str(i)+".png")

			if((joint_or_standard=="joint")):
				show_evolution(loss_segmentation_training,folder_results+"/loss_segmentation/loss_segmentation_training"+str(i)+".png")
				show_evolution(loss_classification_training,folder_results+"/loss_classification/loss_classification_training"+str(i)+".png")
				show_evolution(accuracy_training,folder_results+"/accuracy/training_accuracy"+str(i)+".png")



			if((joint_or_standard=="joint") or (joint_or_standard=="standard")):
				show_evolution(recall_training,folder_results+"/recall/training_recall"+str(i)+".png")
				show_evolution(precision_training,folder_results+"/precision/training_precision"+str(i)+".png")


				for cl in range(nb_classes):
					show_evolution(recall_subclasses_training[cl],folder_results+"/recall_class"+str(cl)+"/training_recall_class"+str(cl)+"_"+str(i)+".png")
					show_evolution(precision_subclasses_training[cl],folder_results+"/precision_class"+str(cl)+"/training_precision_class"+str(cl)+"_"+str(i)+".png")
					show_evolution(dice_subclasses_training[cl],folder_results+"/dice_class"+str(cl)+"/training_dice_class"+str(cl)+"_"+str(i)+".png")

					


			if((joint_or_standard=="classification")):
				show_evolution(accuracy_training,folder_results+"/accuracy/training_accuracy"+str(i)+".png")


			#show_evolution(loss_test,folder_results+"/test_loss"+str(i)+".png")

		if((j>0) and (i>0) and (i%freq_save_parameters==0)):
			net.save_parameters_all_layers(folder_results+"/parameters"+str(i)+"/")
			folder_v=folder_results+"/parameters"+str(i)+"/v"
			write_list_tensors_npy_to_file(v,folder_v)


			#remove previous parameters

			path_previous_parameters=folder_results+"/parameters"+str(i-2*freq_save_parameters)+"/"

			os.system("rm -r "+path_previous_parameters)


			filename_list_loss_training=folder_results+"/parameters"+str(i)+"/loss.txt"
			write_list_to_file(loss_training,filename_list_loss_training)


	print("\n\n\n\n\n\n\nTRAINING FINISHED")
	file_communication=open(filename_communication,"w")
	file_communication.write("finished")
	file_communication.close()

	net.save_parameters_all_layers(folder_results+"/parameters"+str(nb_iterations-1))



	os.system("mkdir -p "+paths_folder_files_launch_tests)

	path_file_input_test=paths_folder_files_launch_tests+"/"+name_test +".txt"

	file_input_test=open(path_file_input_test,"w")

	file_input_test.write(joint_or_standard+" "+name_test+" "+name_net+" "+str(reference_test)+" "+str(num_fold)+" "+type_slice)

	file_input_test.close()

	if(launch_test):
		subprocess.call(["python","launch_all_tests_2D.py",path_file_input_test,folder_all_trainings,folder_results_all_tests])

	



#launch training
list_indexes_start_tmp=[0,0,0]

#train_network(net, lists_images_training, nb_iterations=10001, batch_size=batch_size, nb_batches_one_step=nb_batches_one_step, nb_images_gt_each_batch=nb_images_gt, freq_display=25, freq_compute_stats=25, folder_results=folder_results,list_indexes_start=list_indexes_start_tmp)
train_network(net, lists_images_training, nb_iterations=10001, batch_size=batch_size, nb_batches_one_step=nb_batches_one_step, nb_images_gt_each_batch=nb_images_gt, freq_display=20, freq_compute_stats=20, folder_results=folder_results,list_indexes_start=list_indexes_start_tmp)