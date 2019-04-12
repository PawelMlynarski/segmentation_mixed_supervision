#from theano import function, config, shared, sandbox
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
from theano.ifelse import ifelse
from common_functions_theano import *
from theano.tensor.nnet.bn import batch_normalization






"""
****************
CREATIION OF BATCHES FOR JOINT training
****************
"""





def create_lists_folders_slices_and_files_labels_of_slices(filename_paths_patients_flair):
	file_folders_flair_patients=open(filename_paths_patients_flair,"r")

	list_folders_patients=file_folders_flair_patients.read().split("\n")
	file_folders_flair_patients.close()


	list_files_labels_each_patient=[]

	for folder_flair_patient in list_folders_patients:
		#list_files_labels_each_patient.append(folder_flair_patient.replace("slices_bin_normalized_without_registration/T2_FLAIR/","labels_slices_normalized_without_registration/")+".txt")
		list_files_labels_each_patient.append(folder_flair_patient.replace("slices_bin_normalized_without_registration/T2_FLAIR/","multi_labels_slices_2018/")+".txt")

	return (list_folders_patients,list_files_labels_each_patient)







def choose_images_batch(lists_images_training, list_indexes_start, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt):

	#format of the result: list of elements [path_flair,path_gt,list_present_classes]



	result=[]




	nb_total_positive_cases_with_gt=len(lists_images_training[0])
	nb_total_negative_cases=len(lists_images_training[1])
	nb_total_positive_cases_without_gt=len(lists_images_training[2])



	


	#index for filling the tensors
	num_image=-1


	
	#Remark: if an image is labeled 0, do not read the GT

	for k in range(nb_images_label1_with_gt):
		num_image=num_image+1

		index_image_label1_with_gt=list_indexes_start[0]

		#check if we need to reinitialize the list (put index to 0 and shuffle the list)
		if(index_image_label1_with_gt==nb_total_positive_cases_with_gt):
			list_indexes_start[0]=0
			random.shuffle(lists_images_training[0])
			index_image_label1_with_gt=0

		


		


		element=lists_images_training[0][index_image_label1_with_gt]

		result.append(element)
		



		list_indexes_start[0]=list_indexes_start[0]+1


	for k in range(nb_images_label0):
		num_image=num_image+1

		index_image_label0=list_indexes_start[1]

		#check if we need to reinitialize the list (put index to 0 and shuffle the list)
		if(index_image_label0==nb_total_negative_cases):
			list_indexes_start[1]=0
			random.shuffle(lists_images_training[1])
			index_image_label0=0



		element=lists_images_training[1][index_image_label0]

		result.append(element)


		#Increment the start index
		list_indexes_start[1]=list_indexes_start[1]+1


	for k in range(nb_images_label1_without_gt):
		num_image=num_image+1

		index_image_label1_without_gt=list_indexes_start[2]

		#check if we need to reinitialize the list (put index to 0 and shuffle the list)
		if(index_image_label1_without_gt==nb_total_positive_cases_without_gt):
			list_indexes_start[2]=0
			random.shuffle(lists_images_training[2])
			index_image_label1_without_gt=0

		
	
		element=lists_images_training[2][index_image_label1_without_gt]

		result.append(element)
		#Increment the start index
		list_indexes_start[2]=list_indexes_start[2]+1







	return result








def create_batch_given_list_of_images_batch(list_images_batch, shape_batch, nb_extract_images_label1_with_gt, nb_extract_images_label0, nb_extract_images_label1_without_gt, nb_classes, list_modalities):


	#format of the list_images_batch: list of elements [path_flair,path_gt]
	
	


	batch_size=shape_batch[0]
	height=shape_batch[2]
	width=shape_batch[3]

	nb_modalities=len(list_modalities)

	if(batch_size!=(nb_extract_images_label1_with_gt+nb_extract_images_label0+nb_extract_images_label1_without_gt)):
		print("Problem with the batch size")
		sys.exit(1)

	#initialize tensor
	tensor_images=np.zeros(shape_batch,dtype=theano.config.floatX)
	
	#reminder: nb_images_gt= nb images labeles 1 avec GT + nb images labeles 0

	nb_images_gt=nb_extract_images_label1_with_gt +nb_extract_images_label0

	tensor_labels=np.zeros([nb_images_gt,shape_batch[2],shape_batch[3]],dtype=np.int32)



	list_vectors_all_labels=[]

	#for all tumor subclasses...
	for cl in range(1,nb_classes):
		list_vectors_all_labels.append(np.zeros([batch_size],dtype=np.int32))




	


	#index for filling the tensors
	num_image=-1


	
	#if an image is labeled 0, do not read the GT




	for k in range(nb_extract_images_label1_with_gt):
		num_image=num_image+1


		element=list_images_batch[num_image]
		
		path_flair=element[0]
		path_gt=element[1]
		list_labels_this_slice=element[2]


		for cl in list_labels_this_slice:
			if(cl>0):
				index_class=cl-1
				list_vectors_all_labels[index_class][num_image]=1




				#print("image "+path_flair+" contains the class "+str(cl))


	
		for mod in range(nb_modalities):	
			modality=list_modalities[mod]
			path_one_modality=path_flair.replace("T2_FLAIR",modality)
			tensor_images[num_image, mod, :, :]=np.load(path_one_modality)
		



		#read the GT
		tensor_labels[num_image, :, :]=np.load(path_gt)



	for k in range(nb_extract_images_label0):
		num_image=num_image+1

		element=list_images_batch[num_image]

		path_flair=element[0]
		path_gt=element[1]
		list_labels_this_slice=element[2]


		#check
		for cl in list_labels_this_slice:
			if(cl>0):
				print("Problem: tumor present in a slice supposed to be without tumor")
				sys.exit(2)




		
		for mod in range(nb_modalities):	
			modality=list_modalities[mod]
			path_one_modality=path_flair.replace("T2_FLAIR",modality)
			tensor_images[num_image, mod, :, :]=np.load(path_one_modality)
		
		#don't read the GT: tensor_labels[num_image,:,:] is already at zeros



		


	for k in range(nb_extract_images_label1_without_gt):
		num_image=num_image+1

		
		element=list_images_batch[num_image]


		path_flair=element[0]
		path_gt=element[1]
		list_labels_this_slice=element[2]


		for cl in list_labels_this_slice:
			if(cl>0):
				index_class=cl-1
				list_vectors_all_labels[index_class][num_image]=1


		

		
		for mod in range(nb_modalities):	
			modality=list_modalities[mod]
			path_one_modality=path_flair.replace("T2_FLAIR",modality)
			tensor_images[num_image, mod, :, :]=np.load(path_one_modality)


		
		
		#don't read the GT, we consider that we don't have it (weakly-annotated)

		





	if(num_image!=(batch_size-1)):
		print("\n\n\n\nfonction 'create_tensor': THERE WAS A PROBLEM WITH A NUMBER OF EXTRACTED IMAGES\n\n\n\n")




	#input_batch, tensor_labels_batch, list_vectors_all_labels_batch=create_batch_given_list_of_paths(list_images_batch, batch_shape, nb_images_label1_with_gt, nb_images_label0, nb_images_label1_without_gt, list_modalities)

	return tensor_images,tensor_labels,list_vectors_all_labels








def parse_list_labels2018(list_str,nb_classes,str_region_segmentation):

	list_labels_slice=list_int_from_str(list_str)


	#replace the names of classes


	if(str_region_segmentation=="whole_tumor"):
		for j in range(len(list_labels_slice)):
			#print(list_labels_slice[j])
			if(list_labels_slice[j]>0):
				list_labels_slice[j]=1


	elif(str_region_segmentation=="enhancing_core"):
		for j in range(len(list_labels_slice)):
			if(list_labels_slice[j]==3):
				list_labels_slice[j]=1
			else:
				list_labels_slice[j]=0


	elif(str_region_segmentation=="tumor_core"):
		for j in range(len(list_labels_slice)):
			if((list_labels_slice[j]==3) or (list_labels_slice[j]==1)):
				list_labels_slice[j]=1
			else:
				list_labels_slice[j]=0


	else:
		print("unknown region")
		sys.exit(1)


	#print(list_labels_slice)
	list_labels_slice=list_without_repetitions(list_labels_slice)
	#print(list_labels_slice)

	#print("list_labels_slice")
	#print(list_labels_slice)

	return list_labels_slice









def regroup_labels_one_tensor(input_tensor,str_region_segmentation):

	result_tensor=np.array(input_tensor)


	if(str_region_segmentation=="whole_tumor"):
		result_tensor=(result_tensor>0)

	elif(str_region_segmentation=="tumor_core"):

		result_tensor=np.logical_or((result_tensor==1),(result_tensor==3))

	elif(str_region_segmentation=="enhancing_core"):
		result_tensor=(result_tensor==3)

	else:
		print("unknown region")
		sys.exit(1)


	#from boolean to int
	result_tensor=np.array(result_tensor,dtype=input_tensor.dtype)


	

	return result_tensor






















def create_lists_images_training(list_folders_flair_patients_training_with_gt,list_files_labels_each_training_patient_with_gt,list_folders_flair_patients_training_without_gt,list_files_labels_each_training_patient_without_gt, slice_start, slice_end,nb_classes,str_region_segmentation):
	#we want 3 lists: [paths to flair 2D images labeled 1 for which we have the GT, paths to flair 2D images labeled 0, paths to flair 2D images labeled 1 for which we don't have the GT]
	#for other functions: if we have a GT for one 2D image, to access to it, we only have to replace 'T2_FLAIR' by 'GT' in the path

	result=[[],[],[]]


	k=-1
	for folder_patient_with_gt in list_folders_flair_patients_training_with_gt:
		k=k+1

		#read the labels of slices



		file_labels_one_patient=open(list_files_labels_each_training_patient_with_gt[k],"r")
		list_labels_one_patient=file_labels_one_patient.read().split("\n")
		file_labels_one_patient.close()





		#print("\n\n\n\n\n\n\n\nlist labels of the patient "+folder_patient_with_gt)
		#print(list_labels_one_patient)
		#add the slices of this patient to the result

		for s in range(slice_start,slice_end+1):






			list_labels_this_slice=parse_list_labels2018(list_labels_one_patient[s].split(","),nb_classes,str_region_segmentation)
		
		

			if((len(list_labels_this_slice)==1) and (list_labels_this_slice[0]==0)):
				label_this_slice=0
			else:
				label_this_slice=1

				#print(list_labels_this_slice)



			path_flair=folder_patient_with_gt+"/slice"+str(s)+".npy"


			path_gt=path_flair.replace("T2_FLAIR","GT")




			element=[path_flair,path_gt,list_labels_this_slice]



			if(label_this_slice>0):
				element=[path_flair,path_gt,list_labels_this_slice]
				#positive slice with GT
				result[0].append(element)
			else:
				#negative slice
				result[1].append(element)
			







	k=-1
	for folder_patient_without_gt in list_folders_flair_patients_training_without_gt:
		k=k+1


		file_labels_one_patient=open(list_files_labels_each_training_patient_without_gt[k],"r")
		list_labels_one_patient=file_labels_one_patient.read().split("\n")
		file_labels_one_patient.close()



		for s in range(slice_start,slice_end+1):

			#read the labels of slices

			list_labels_this_slice=parse_list_labels2018(list_labels_one_patient[s].split(","),nb_classes,str_region_segmentation)



			if((len(list_labels_this_slice)==1) and (list_labels_this_slice[0]==0)):
				label_this_slice=0
			else:
				label_this_slice=1



			path_flair=folder_patient_without_gt+"/slice"+str(s)+".npy"


			path_gt=path_flair.replace("T2_FLAIR","GT")

			

			element=[path_flair,path_gt,list_labels_this_slice]

			


			if(label_this_slice>0):
				element=[path_flair,path_gt,list_labels_this_slice]
				#positive slice with GT
				result[2].append(element)
			else:
				#negative slice
				result[1].append(element)
			

	

	#randomly shuffle the lists (to avoid overfitting: similar slices)
	

	random.shuffle(result[0])
	random.shuffle(result[1])
	random.shuffle(result[2])



	print("\n\n\nThere are "+str(len(result[0]))+" images labeled 1 with GT, "+str(len(result[1]))+" images labeled 0, "+str(len(result[2]))+" images labeled 1 without GT")
	print("\n\n\n")

	#looks correct

	return result








def generate_axial_flips_batch_and_gt(input_4Dbatch, input_3D_GT, proportion_flip):
	#results: 4D tensor flipped_batch, 3D integer tensor flipped_GT
	result_batch=np.array(input_4Dbatch,dtype=theano.config.floatX)
	result_GT=np.array(input_3D_GT,dtype=np.int32)
	

	nb_images=input_4Dbatch.shape[0]
	#print("nb images="+str(nb_images))
	nb_images_gt=input_3D_GT.shape[0]

	nb_images_flip=int(round(proportion_flip * float(nb_images)))
	
	indexes=range(nb_images)

	random.shuffle(indexes)

	#print("flip "+str(nb_images_flip)+" images")
	for k in range(nb_images_flip):
		
		index_flip=indexes[k]
		#print("flipping image "+str(index_flip))
		#flip the image at index_flip
		result_batch[index_flip,:,:,:]=input_4Dbatch[index_flip,:,:,::-1]
		if(index_flip<nb_images_gt):
			result_GT[index_flip,:,:]=input_3D_GT[index_flip,:,::-1]


	return (result_batch,result_GT)







"""
****************
CREATIION OF BATCHES FOR STANDARD training
****************
"""














"""
****************************
ANALYSIS/PARSING OF FILES
****************************
"""









class layer_definition(object):
	def __init__(self):
		self.name=None
		self.type=None
		self.activation=None
		self.kernel_size=None
		self.stride=None
		self.nb_outputs=None
		self.layer1=""
		self.layer2=""
		self.bottom_layer=""
		self.channel_start=0
		self.channel_end=3
		self.weight_loss=1.0





def analyze_network(filename_network_definition):
	list_names=[]
	list_types=[]
	list_layers=[]
	file_network=open(filename_network_definition,"r")
	content_file=file_network.read()

	list_str_layers=content_file.split("layer")
	#ignore the first element of the list (not a layer definition)
	i=-1


	new_layer=layer_definition()

	for k in range(1,len(list_str_layers)):
		content_layer=list_str_layers[k]

		lines_layer=content_layer.split("\n")
		#attention: there can be subdefinitions of some compontents of the layer, for example of the "weight filler"
		name_known=False
		type_known=False
		name_layer=""
		pool_type=""
		stride=1
		upsampling_ratio=1
		kernel_size=-1
		num_output=0
		channel_start=-1
		channel_end=-1
		weight_loss=1.0


		class_loss=-1

		concatenation_layer1=""
		concatenation_layer2=""
		bottom_layer=""

		for line in lines_layer:
			if((not(name_known)) and ("name:" in line)):
				name_layer=retrieve_argument_line_protobuf(line).replace("\"","")
				name_known=True

			if((not(type_known)) and ("type:" in line)):
				type_layer=retrieve_argument_line_protobuf(line).replace("\"","")
				type_known=True



			if(("pool:" in line)):
				pool_type=retrieve_argument_line_protobuf(line)
				#print("\nPOOL")
				#print(pool_type)

			if(("stride:" in line)):
				stride=int(retrieve_argument_line_protobuf(line))

			if(("num_output:" in line)):
				num_output=int(retrieve_argument_line_protobuf(line))

			if(("kernel_size:" in line)):
				kernel_size=int(retrieve_argument_line_protobuf(line))


			if(("channel_start:" in line)):
				channel_start=int(retrieve_argument_line_protobuf(line))

			if(("channel_end:" in line)):
				channel_end=int(retrieve_argument_line_protobuf(line))


			if(("weight_loss:" in line)):
				weight_loss=float(retrieve_argument_line_protobuf(line))



			if(("class_loss:" in line)):
				class_loss=int(retrieve_argument_line_protobuf(line))


			if(("bottom:" in line)):
				bottom_layer=retrieve_argument_line_protobuf(line).replace('"','')






			if(("upsampling_ratio:" in line)):
				upsampling_ratio=int(retrieve_argument_line_protobuf(line))
			#print(line)
			if(("input1:" in line)):
				#print("\n\n\nICCCCIIIII\n\n\n\n\n")
				concatenation_layer1=retrieve_argument_line_protobuf(line)

			if(("input2:" in line)):
				concatenation_layer2=retrieve_argument_line_protobuf(line)

		if((type_layer=="Accuracy") or (type_layer=="SoftmaxWithLoss")):
			continue


		#print("iiiii="+str(i))
		#print("\n\nTYPE LAYER: "+type_layer)
		if((type_layer=="Data") or (type_layer=="Convolution") or (type_layer=="Pooling") or (type_layer=="FC") or (type_layer=="BilinearUpsampling") or (type_layer=="Concatenation") or (type_layer=="ConcatenationFC") or (type_layer=="loss_segmentation") or (type_layer=="loss_classification")):
			list_names.append(name_layer)
			new_layer=layer_definition()
			new_layer.name=name_layer
			new_layer.activation="none"

			if(type_layer!="Data"):
				new_layer.bottom_layer=bottom_layer
			list_layers.append(new_layer)

			i=i+1
			#print("iiiii="+str(i))

			#reminder: we precise the type of the layer, the activation and the stride


			if(type_layer=="loss_segmentation"):
				#list_types.append(["fc","none"])
				new_layer.type="loss_segmentation"
				#list_types.append(["loss","none"])
				i=i-1
				new_layer.weight_loss=weight_loss


			if(type_layer=="loss_classification"):
				#list_types.append(["fc","none"])
				new_layer.type="loss_classification"
				#list_types.append(["loss","none"])
				i=i-1
				new_layer.class_loss=class_loss



			if(type_layer=="FC"):
				list_types.append(["fc","none"])
				new_layer.type="fc"
				new_layer.nb_outputs=num_output


			if((type_layer=="Data")):
				list_types.append(["data","none"])
				new_layer.type="data"
				new_layer.channel_start=channel_start
				new_layer.channel_end=channel_end
				

			if((type_layer=="BilinearUpsampling")):
				list_types.append(["bilinear_upsampling","none"])
				new_layer.type="bilinear_upsampling"
				new_layer.upsampling_ratio=upsampling_ratio

			if((type_layer=="Concatenation")):
				list_types.append(["concatenation","none"])
				new_layer.type="concatenation"
				#print("\n\n\nCONCATENATION LAYER1: "+concatenation_layer1)
				new_layer.layer1=concatenation_layer1
				new_layer.layer2=concatenation_layer2



			if((type_layer=="ConcatenationFC")):
				list_types.append(["concatenation_fc","none"])
				new_layer.type="concatenation_fc"
				#print("\n\n\nCONCATENATION LAYER1: "+concatenation_layer1)
				new_layer.layer1=concatenation_layer1
				new_layer.layer2=concatenation_layer2


			if((type_layer=="Convolution")):


				list_types.append(["conv","none",stride])
				new_layer.type="conv"
				new_layer.stride=stride
				new_layer.kernel_size=[kernel_size,kernel_size]
				new_layer.nb_outputs=num_output


				#print("convolution "+new_layer.name)
				#print(list_types[-1])

				#print("len "+str(len(list_types)))
				#print("i "+str(i))

			#print("\n\ntype_layer: "+type_layer)
			#print("\n\npool: "+pool_type)
			if((type_layer=="Pooling") and (pool_type=="MAX")):
				list_types.append(["max_pooling",[kernel_size,kernel_size],stride])
				new_layer.type="max_pooling"
				new_layer.kernel_size=[kernel_size,kernel_size]
				new_layer.stride=stride




			if((type_layer=="Pooling") and (pool_type=="MEAN")):
				list_types.append(["mean_pooling",[kernel_size,kernel_size],stride])
				new_layer.type="mean_pooling"
				new_layer.kernel_size=[kernel_size,kernel_size]
				new_layer.stride=stride

		#elif(type_layer!="ReLU"):
		#	print("typeee "+type_layer)
			
		if((type_layer=="ReLU")):
			#set the activation of the previous layer 
			
			list_types[i][1]="relu"
			new_layer.activation="relu"
			#print("apres")
			#print(list_types[i-1])
		if((type_layer=="Sigmoid")):
			list_types[i][1]="sigmoid"
			new_layer.activation="sigmoid"
		if((type_layer=="TanH")):
			list_types[i][1]="tanh"
			new_layer.activation="tanh"


	file_network.close()
	return list_layers
	#return [list_names,list_types]
	



def find_name_last_conv_layer(list_layers_definition):
	result=""
	for k in range(len(list_layers_definition)):
		if(list_layers_definition[k].type=="conv"):
			result=list_layers_definition[k].name




	print("Name of the last conv layer:"+result)
	return result





def retrieve_argument_line_protobuf(line):
	tmp=line.split(": ")
	return tmp[1]



def list_int(list_input):
	result=[]
	for el in list_input:
		result.append(int(el))
	return result

def list_float(list_input):
	result=[]
	for el in list_input:
		result.append(float(el))
	return result

def print_2D_matrix(matrix_np):
	for y in range(matrix_np.shape[0]):
		#if(y>0):
		#print("\n")
		#print(matrix_np[y,:])
		print("["+" ".join(str(s) for s in matrix_np[y,:])+"]")







"""
****************************
LAYERS
****************************
"""






class Data_Layer2D(object):
	def __init__(self, str_name, channel_start, channel_end, input_tensor_shape, expr_output_previous_layer):
		self.name=str_name
		self.type="data"
		self.bottom_layer=""
		self.channel_start=channel_start
		self.channel_end=channel_end
		self.output=expr_output_previous_layer[:,channel_start:(channel_end+1),:,:]
		self.output_shape=[input_tensor_shape[0], channel_end-channel_start+1,input_tensor_shape[2],input_tensor_shape[3]]
		self.factor_y=1
		self.factor_x=1
		self.total_offset_y=0
		self.total_offset_x=0


class FC_Layer2D(object):
	def __init__(self, str_name, nb_outputs, str_activation, input_layer):
		"""
		Constructor of the class: define the mathematical expression and initialize parameters
		input_layer.output= symbolic Theano 2D tensor (indexes: index_image, index_in_the_flatten_multichannel_image)
		an object of this class has to contain a mathematical expression of the output of a fully-connected layer
		"""

		self.name=str_name
		self.type="fc"
		self.nb_outputs=nb_outputs
		self.activation=str_activation
		self.bottom_layer=input_layer.name
		
	


		if(input_layer.type=="fc"):
			shape_input=input_layer.output_shape
		elif(input_layer.type=="concatenation_fc"):
			shape_input=input_layer.output_shape

		else:
			shape_input=[input_layer.output_shape[0],input_layer.output_shape[1]*input_layer.output_shape[2]*input_layer.output_shape[3]]



		nb_input_neurons=shape_input[1]


		
		self.w_shape=[nb_input_neurons,nb_outputs]



		#He initialization
		if(str_activation=="relu"):
			gain=2.0
		else:
			gain=1.0
		std_init=np.sqrt(gain)*np.sqrt(1.0/float(nb_input_neurons))
		self.W=theano.shared(np.asarray(np.random.normal(loc=0.0, scale=std_init, size=self.w_shape),dtype=theano.config.floatX), borrow=True)
		

	

		self.b=theano.shared(np.zeros([nb_outputs],dtype=theano.config.floatX), borrow=True)
		


		self.w_norm2_2=T.sum(self.W**2)
		self.parameters=[self.W, self.b]



		
		#self.output_shape=[shape_input[0], nb_outputs]

		self.output_shape=[shape_input[0], nb_outputs]


		if(input_layer.type=="fc"):
			output_fc=T.dot(input_layer.output,self.W)+self.b
		else:
			output_fc=T.dot((input_layer.output).flatten(2),self.W)+self.b
		
		if(str_activation=="relu"):
			self.output=T.nnet.relu(output_fc)
		elif(str_activation=="sigmoid"):
			self.output=T.nnet.sigmoid(output_fc)
		elif(str_activation=="tanh"):
			self.output=T.tanh(output_fc)
		else:
			#no activation
			self.output=output_fc


		
	def set_parameters(self, folder_parameters):
		
		filename_w=folder_parameters+"/"+self.name+"_w.npy"
		filename_b=folder_parameters+"/"+self.name+"_b.npy"

		
		w_values_array=np.load(filename_w)
		b_values_array=np.load(filename_b)

		self.W.set_value(w_values_array)
		self.b.set_value(b_values_array)
		self.w_norm2_2=T.sum(self.W**2)







	def print_parameters(self):
		w=self.W.get_value()
		b=self.b.get_value()
		print("\n\n\n\n\n\n\n\n\n\n\nFully-connected layer "+self.name)
		print("Parameters W (first neuron):")
		print("Weights of the first neuron:")
		print(w[0][:])
		print("Parameter b (first neuron):"+str(b[0]))





	def save_parameters(self, output_folder_parameters):
		w=self.W.get_value()
		b=self.b.get_value()



		filename_w=output_folder_parameters+"/"+self.name+"_w.npy"


		filename_b=output_folder_parameters+"/"+self.name+"_b.npy"

		np.save(filename_w, w)
		np.save(filename_b, b)
	











class FC_Layer2D_bn(object):
	def __init__(self, str_name, nb_outputs, str_activation, input_tensor_shape, input_to_layer, filename_mean_and_std_layer=None):
		"""
		Constructor of the class: define the mathematical expression and initialize parameters
		input_layer.output= symbolic Theano 2D tensor (indexes: index_image, index_in_the_flatten_multichannel_image)
		an object of this class has to contain a mathematical expression of the output of a fully-connected layer
		"""

		self.mean_fixed=None
		self.std_fixed=None

		self.name=str_name
		self.type="fc"
		self.nb_outputs=nb_outputs

		self.bottom_layer=input_layer.name


		if(filename_mean_and_std_layer!=None):
			#apply the functions which creates Theano shared variables and reads the values from a file
			self.set_fixed_mean_and_std(filename_mean_and_std_layer)



		nb_input_neurons=input_tensor_shape[1]


		nb_neurons_next_layer=200
		


		W_bound=np.sqrt(6.0/(nb_input_neurons+nb_neurons_next_layer))
		self.activation=str_activation


		#self.w_shape=[nb_outputs,nb_input_neurons]
		self.w_shape=[nb_input_neurons,nb_outputs]
		#rng = np.random.RandomState(1234)
		
		"""
		rng = np.random.RandomState()
		self.W=theano.shared(np.asarray(rng.uniform(low=-W_bound,high=W_bound, size=self.w_shape),dtype=theano.config.floatX), borrow=True)
		"""




		#He initialization
		if(str_activation=="relu"):
			gain=2.0
		else:
			gain=1.0
		std_init=np.sqrt(gain)*np.sqrt(1.0/float(nb_input_neurons))
		self.W=theano.shared(np.asarray(np.random.normal(loc=0.0, scale=std_init, size=self.w_shape),dtype=theano.config.floatX), borrow=True)
		





		self.b=theano.shared(np.zeros([nb_outputs],dtype=theano.config.floatX), borrow=True)
		self.gamma=theano.shared(np.ones([nb_outputs],dtype=theano.config.floatX), borrow=True)
		self.beta=theano.shared(np.zeros([nb_outputs],dtype=theano.config.floatX), borrow=True)
		self.parameters=[self.W, self.b , self.gamma, self.beta]


		self.w_norm2_2=T.sum(self.W**2)
	
		self.output_shape=[input_tensor_shape[0], nb_outputs]
		
		fc_output=T.dot(input_to_layer,self.W)+self.b

		if(self.mean_fixed!=None):
			self.mean=self.mean_fixed
			self.std=self.std_fixed
		else:
			self.mean=fc_output.mean(axis=[0])
			self.std=fc_output.std(axis=[0])

		if(str_activation=="relu"):
			self.output=T.nnet.relu(batch_normalization(fc_output, self.gamma, self.beta, self.mean, self.std, mode="high_mem"))
		elif(str_activation=="sigmoid"):
			self.output=T.nnet.sigmoid(batch_normalization(fc_output, self.gamma, self.beta, self.mean, self.std, mode="high_mem"))
		elif(str_activation=="tanh"):
			self.output=T.tanh(batch_normalization(fc_output, self.gamma, self.beta, self.mean, self.std, mode="high_mem"))
		else:
			#no activation
			self.output=batch_normalization(fc_output, self.gamma, self.beta, self.mean, self.std, mode="high_mem")


		


	def set_parameters(self, folder_parameters):
		
		filename_w=folder_parameters+"/"+self.name+"_w.npy"
		filename_b=folder_parameters+"/"+self.name+"_b.npy"
		filename_gamma=folder_parameters+"/"+self.name+"_gamma.npy"
		filename_beta=folder_parameters+"/"+self.name+"_beta.npy"

		
		w_values_array=np.load(filename_w)
		b_values_array=np.load(filename_b)
		gamma_values_array=np.load(filename_gamma)
		beta_values_array=np.load(filename_beta)

		self.W.set_value(w_values_array)
		self.b.set_value(b_values_array)
		self.gamma.set_value(gamma_values_array)
		self.beta.set_value(beta_values_array)

		self.w_norm2_2=T.sum(self.W**2)




	def set_fixed_mean_and_std(self, filename_mean_and_std_layer):
		#read the file with means and standard deviations (before activation function)
		#Format of the file: line 1: length of vectors with means and standard deviations, line 2=means of each feature map, line 3=standard deviations


		#create Theano shared variables
		self.mean_fixed=theano.shared(np.zeros([self.nb_outputs],dtype=theano.config.floatX), borrow=True)
		self.std_fixed=theano.shared(np.zeros([self.nb_outputs],dtype=theano.config.floatX), borrow=True)

		#read the values from the files


		txt_mean_and_std=open(filename_mean_and_std_layer,"r").read()
		lines=txt_mean_and_std.split("\n")
		dim=int(lines[0])
	
		
		if(dim!=self.nb_outputs):
			print("\n\n\n\n\n\n\nPROBLEM WITH READING MEAN/STANDARD DEVIATION OF THE CONVOLUTIONAL LAYER: DIMENSIONS MISMATCH")
			print("expected:")
			print(self.nb_outputs)
			print("found:")
			print(dim)
			sys.exit(3)


		

		mean_values_file=list_float(lines[1].split(" "))
		mean_values_array=np.zeros([self.nb_outputs],dtype=theano.config.floatX)
		for num_output_channel in range(self.nb_outputs):
			mean_values_array[num_output_channel]=mean_values_file[num_output_channel]




		std_values_file=list_float(lines[2].split(" "))
		std_values_array=np.zeros([self.nb_outputs],dtype=theano.config.floatX)
		for num_output_channel in range(self.nb_outputs):
			std_values_array[num_output_channel]=std_values_file[num_output_channel]




		
		self.mean_fixed.set_value(mean_values_array)
		self.std_fixed.set_value(std_values_array)



	def print_parameters(self):
		w=self.W.get_value()
		b=self.b.get_value()
		gamma=self.gamma.get_value()
		beta=self.beta.get_value()
		print("\n\n\n\n\n\n\n\n\n\n\nFully-connected layer "+self.name)
		print("Parameters W (first neuron):")
		print("Weights of the first neuron:")
		print(w[0][:])
		print("Parameter b (first neuron):"+str(b[0]))
		print("Parameter gamma (first filter):"+str(gamma[0]))
		print("Parameter beta (first filter):"+str(beta[0]))







	def save_parameters(self, output_folder_parameters):
		w=self.W.get_value()
		b=self.b.get_value()
		gamma=self.gamma.get_value()
		beta=self.beta.get_value()


		filename_w=output_folder_parameters+"/"+self.name+"_w.npy"
		filename_b=output_folder_parameters+"/"+self.name+"_b.npy"
		filename_gamma=output_folder_parameters+"/"+self.name+"_gamma.npy"
		filename_beta=output_folder_parameters+"/"+self.name+"_beta.npy"


		np.save(filename_w, w)
		np.save(filename_b, b)
		np.save(filename_gamma, gamma)
		np.save(filename_beta, beta)

	


class Max_Pooling_Layer2D(object):
 	def __init__(self, str_name, kernel_size, strides, input_layer):
		"""
		Constructor of the class: define the mathematical expression
		input_layer.output= symbolic Theano 4D tensor (indexes: image, input_channel, y, x)
		an object of this class has to contain a mathematical expression of the output of a max-pooling layer
		remark: if the previous layer is a 4d-tensor, apply flatten(2) to the input 
		"""

		self.type="max_pooling"
		self.bottom_layer=input_layer.name
		self.name=str_name
		self.strides=strides
		self.kernel_shape=kernel_size

		input_tensor_shape=input_layer.output_shape

		
		if(kernel_size[0]%2==0):
			#pair
			self.offset_y=(kernel_size[0]/2)-1
		else:
			#odd
			self.offset_y=kernel_size[0]/2

		if(kernel_size[1]%2==0):
			#pair
			self.offset_x=(kernel_size[1]/2)-1
		else:
			#odd
			self.offset_x=kernel_size[1]/2




		#downsampling/upsampling ratio (1= no downsampling)
		self.factor_y=input_layer.factor_y*self.strides[0]
		self.factor_x=input_layer.factor_x*self.strides[1]



		self.total_offset_y=input_layer.total_offset_y+ self.offset_y*(input_layer.factor_y)
		self.total_offset_x=input_layer.total_offset_x+ self.offset_x*(input_layer.factor_x)


		height_input=input_tensor_shape[2]
		width_input=input_tensor_shape[3]
		

		#case with 'ignore_border=False'. If 'ignore_border=True', do not take the pool if it strictly exceeds the border
		number_pools_y=0
		for z in range(height_input):
			index=z*strides[0]
			if(index<height_input):
				chosen_y=index
				number_pools_y=number_pools_y+1

			if((index+(kernel_size[0]-1))>=(height_input-1)):
				#first y which reaches the end
				chosen_y=index
				break


		number_pools_x=0
		for z in range(width_input):
			index=z*strides[1]
			if(index<width_input):
				chosen_x=index
				number_pools_x=number_pools_x+1
			if((index+(kernel_size[1]-1))>=(width_input-1)):
				#first x which reaches the end
				chosen_x=index
				break



		#height_output=(input_tensor_shape[2]-1)/strides[0]+1
		#width_output=(input_tensor_shape[3]-1)/strides[1]+1
		height_output=number_pools_y
		width_output=number_pools_x

		self.output_shape=[input_tensor_shape[0],input_tensor_shape[1],height_output,width_output]

		#self.output=theano.tensor.signal.pool.pool_2d(input_layer.output, ds=kernel_size,st=strides, ignore_border=False)
		self.output=theano.tensor.signal.pool.pool_2d(input_layer.output, ws=kernel_size,stride=strides, ignore_border=False)
	













class Mean_Pooling_Layer2D(object):
 	def __init__(self, str_name, kernel_size, strides, input_layer):
		"""
		Constructor of the class: define the mathematical expression
		input_layer.output= symbolic Theano 4D tensor (indexes: image, input_channel, y, x)
		an object of this class has to contain a mathematical expression of the output of a max-pooling layer
		remark: if the previous layer is a 4d-tensor, apply flatten(2) to the input 
		"""

		self.type="mean_pooling"
		self.bottom_layer=input_layer.name
		self.name=str_name
		self.strides=strides
		self.kernel_shape=kernel_size

		input_tensor_shape=input_layer.output_shape

		
		if(kernel_size[0]%2==0):
			#pair
			self.offset_y=(kernel_size[0]/2)-1
		else:
			#odd
			self.offset_y=kernel_size[0]/2

		if(kernel_size[1]%2==0):
			#pair
			self.offset_x=(kernel_size[1]/2)-1
		else:
			#odd
			self.offset_x=kernel_size[1]/2




		#downsampling/upsampling ratio (1= no downsampling)
		self.factor_y=input_layer.factor_y*self.strides[0]
		self.factor_x=input_layer.factor_x*self.strides[1]



		self.total_offset_y=input_layer.total_offset_y+ self.offset_y*(input_layer.factor_y)
		self.total_offset_x=input_layer.total_offset_x+ self.offset_x*(input_layer.factor_x)


		height_input=input_tensor_shape[2]
		width_input=input_tensor_shape[3]
		

		#case with 'ignore_border=False'. If 'ignore_border=True', do not take the pool if it strictly exceeds the border
		number_pools_y=0
		for z in range(height_input):
			index=z*strides[0]
			if(index<height_input):
				chosen_y=index
				number_pools_y=number_pools_y+1

			if((index+(kernel_size[0]-1))>=(height_input-1)):
				#first y which reaches the end
				chosen_y=index
				break


		number_pools_x=0
		for z in range(width_input):
			index=z*strides[1]
			if(index<width_input):
				chosen_x=index
				number_pools_x=number_pools_x+1
			if((index+(kernel_size[1]-1))>=(width_input-1)):
				#first x which reaches the end
				chosen_x=index
				break



		#height_output=(input_tensor_shape[2]-1)/strides[0]+1
		#width_output=(input_tensor_shape[3]-1)/strides[1]+1
		height_output=number_pools_y
		width_output=number_pools_x

		self.output_shape=[input_tensor_shape[0],input_tensor_shape[1],height_output,width_output]

		#self.output=theano.tensor.signal.pool.pool_2d(input_layer.output, ds=kernel_size,st=strides, ignore_border=False)
		self.output=theano.tensor.signal.pool.pool_2d(input_layer.output, ws=kernel_size,stride=strides, ignore_border=False, mode='average_inc_pad')
	















class ConvLayer2D_bn(object):
	def __init__(self, str_name, kernel_dim, strides, nb_output_feature_maps,  str_activation, input_layer, filename_mean_and_std_layer=None,apply_bn=True,nb_images_compute_output=-1):
		"""
		Constructor of the class: define the mathematical expression and initialize parameters

		kernel_dim=(kernel_height, kernel_width)
		input_layer.output= symbolic Theano 4D tensor (indexes: image, input_channel, y, x)
		an object of this class has to contain a mathematical expression of the output of a convolutional layer
		"""
		#input_tensor_shape=input_layer.output.shape
		#print("INPUT SHAPE")
		#print(input_tensor_shape)
	
		self.mean_fixed=None
		self.std_fixed=None


		self.apply_bn=apply_bn


		if(nb_images_compute_output==(-1)):
			#standard case
			input_tensor_shape=input_layer.output_shape
			input_to_layer=input_layer.output

		else:
			#case of segmentation layers: compute only the outputs for images with the GT
			input_tensor_shape=[nb_images_compute_output,input_layer.output_shape[1],input_layer.output_shape[2],input_layer.output_shape[3]]
			input_to_layer=input_layer.output[:nb_images_compute_output,:,:,:]


		self.name=str_name
		self.type="conv"
		self.bottom_layer=input_layer.name
	
		if(apply_bn):
			print("Layer "+self.name+": apply batch normalization")
		else:
			print("Layer "+self.name+": no batch normalization")


		#self.strides=[1,1]
		self.strides=strides
		self.nb_outputs=nb_output_feature_maps


		if(apply_bn and (filename_mean_and_std_layer!=None)):
			#apply the functions which creates Theano shared variables and reads the values from a file
			self.set_fixed_mean_and_std(filename_mean_and_std_layer)



		if(kernel_dim[0]%2==0):
			#pair
			self.offset_y=(kernel_dim[0]/2)-1
		else:
			#odd
			self.offset_y=kernel_dim[0]/2

		if(kernel_dim[1]%2==0):
			#pair
			self.offset_x=(kernel_dim[1]/2)-1
		else:
			#odd
			self.offset_x=kernel_dim[1]/2



		#downsampling/upsampling ratio (1= no downsampling)
		self.factor_y=input_layer.factor_y*self.strides[0]
		self.factor_x=input_layer.factor_x*self.strides[1]



		self.total_offset_y=input_layer.total_offset_y+ self.offset_y*(input_layer.factor_y)
		self.total_offset_x=input_layer.total_offset_x+ self.offset_x*(input_layer.factor_x)




		#random initialization of the kernels
		#rng = np.random.RandomState(1234)
		rng = np.random.RandomState()
		nb_input_feature_maps=input_tensor_shape[1]
	
		neurons_in=kernel_dim[0]*kernel_dim[1]*nb_input_feature_maps



		neurons_out=kernel_dim[0]*kernel_dim[1]*nb_output_feature_maps








		W_bound=np.sqrt(6.0/(neurons_in+neurons_out))
		self.activation=str_activation

		self.w_shape=[nb_output_feature_maps,nb_input_feature_maps,kernel_dim[0],kernel_dim[1]]

		#self.W=theano.shared(np.asarray(rng.uniform(low=-W_bound,high=W_bound, size=self.w_shape),dtype=theano.config.floatX), borrow=True)
		
		
		#He initialization
		if(str_activation=="relu"):
			gain=2.0
		else:
			gain=1.0
		std_init=np.sqrt(gain)*np.sqrt(1.0/float(neurons_in))
		self.W=theano.shared(np.asarray(np.random.normal(loc=0.0, scale=std_init, size=self.w_shape),dtype=theano.config.floatX), borrow=True)
		


		self.b=theano.shared(np.zeros([nb_output_feature_maps],dtype=theano.config.floatX), borrow=True)

		self.gamma=theano.shared(np.ones([nb_output_feature_maps],dtype=theano.config.floatX), borrow=True)
		self.beta=theano.shared(np.zeros([nb_output_feature_maps],dtype=theano.config.floatX), borrow=True)

		if(apply_bn):
			self.parameters=[self.W, self.b , self.gamma, self.beta]
		else:
			self.parameters=[self.W, self.b]

		

		self.w_norm2_2=T.sum(self.W**2)
		#construct the mathematical expression



		#determine the size of the output

		height_input=input_tensor_shape[2]
		width_input=input_tensor_shape[3]
		number_pools_y=0
		
		for z in range(height_input):
			index=z*strides[0]
			if((index+(kernel_dim[0]-1))<=(height_input-1)):
				#if the index is compatible
				chosen_y=index
				number_pools_y=number_pools_y+1
			else:
				break


		number_pools_x=0
		
		for z in range(width_input):
			index=z*strides[1]
			if((index+(kernel_dim[1]-1))<=(width_input-1)):
				#if the index is compatible
				chosen_x=index
				number_pools_x=number_pools_x+1
			else:
				break


		height_output=number_pools_y
		width_output=number_pools_x


		self.output_shape=[input_tensor_shape[0],nb_output_feature_maps,height_output,width_output]

	

		conv_output=T.nnet.conv2d(input=input_to_layer,filters=self.W,filter_shape=self.w_shape, input_shape=input_tensor_shape)+self.b.dimshuffle('x',0,'x','x')



		if(self.mean_fixed!=None):
			self.mean=self.mean_fixed.dimshuffle('x',0,'x','x')
			self.std=self.std_fixed.dimshuffle('x',0,'x','x')
		else:
			self.mean=conv_output.mean(axis=[0,2,3]).dimshuffle('x',0,'x','x')
			self.std=conv_output.std(axis=[0,2,3]).dimshuffle('x',0,'x','x')


		if(apply_bn):
			g=self.gamma.dimshuffle('x',0,'x','x')
			beta=self.beta.dimshuffle('x',0,'x','x')
			bn_conv_output=batch_normalization(conv_output, g, beta, self.mean, self.std, mode="high_mem")

		else:
			g=None
			beta=None
			bn_conv_output=None




		



		if(str_activation=="relu"):
			if(apply_bn):
				self.output=T.nnet.relu(bn_conv_output)
			else:
				print("Ici le layer "+self.name+", je n'utilise pas BN")
				self.output=T.nnet.relu(conv_output)

		elif(str_activation=="sigmoid"):
			if(apply_bn):
				self.output=T.nnet.sigmoid(bn_conv_output)
			else:
				print("Ici le layer "+self.name+", je n'utilise pas BN")
				self.output=T.nnet.sigmoid(conv_output)


		elif(str_activation=="tanh"):
			if(apply_bn):
				self.output=T.tanh(bn_conv_output)
			else:
				print("Ici le layer "+self.name+", je n'utilise pas BN")
				self.output=T.tanh(conv_output)

		else:
			#no activation
			if(apply_bn):
				self.output=bn_conv_output
			else:
				print("Ici le layer "+self.name+", je n'utilise pas BN")
				self.output=conv_output








	def set_parameters(self, folder_parameters):
		
		filename_w=folder_parameters+"/"+self.name+"_w.npy"
		filename_b=folder_parameters+"/"+self.name+"_b.npy"
		filename_gamma=folder_parameters+"/"+self.name+"_gamma.npy"
		filename_beta=folder_parameters+"/"+self.name+"_beta.npy"

		
		w_values_array=np.load(filename_w)
		b_values_array=np.load(filename_b)
		gamma_values_array=np.load(filename_gamma)
		beta_values_array=np.load(filename_beta)

		self.W.set_value(w_values_array)
		self.b.set_value(b_values_array)
		self.gamma.set_value(gamma_values_array)
		self.beta.set_value(beta_values_array)

		self.w_norm2_2=T.sum(self.W**2)









	def set_fixed_mean_and_std(self, filename_mean_and_std_layer):
		#read the file with means and standard deviations (before activation function)
		#Format of the file: line 1: length of vectors with means and standard deviations, line 2=means of each feature map, line 3=standard deviations


		#create Theano shared variables
		self.mean_fixed=theano.shared(np.zeros([self.nb_outputs],dtype=theano.config.floatX), borrow=True)
		self.std_fixed=theano.shared(np.zeros([self.nb_outputs],dtype=theano.config.floatX), borrow=True)

		#read the values from the files


		txt_mean_and_std=open(filename_mean_and_std_layer,"r").read()
		lines=txt_mean_and_std.split("\n")
		dim=int(lines[0])
	
		
		if(dim!=self.nb_outputs):
			print("\n\n\n\n\n\n\nPROBLEM WITH READING MEAN/STANDARD DEVIATION OF THE CONVOLUTIONAL LAYER: DIMENSIONS MISMATCH")
			print("expected:")
			print(self.nb_outputs)
			print("found:")
			print(dim)
			sys.exit(3)


		

		mean_values_file=list_float(lines[1].split(" "))
		mean_values_array=np.zeros([self.nb_outputs],dtype=theano.config.floatX)
		for num_output_channel in range(self.nb_outputs):
			mean_values_array[num_output_channel]=mean_values_file[num_output_channel]




		std_values_file=list_float(lines[2].split(" "))
		std_values_array=np.zeros([self.nb_outputs],dtype=theano.config.floatX)
		for num_output_channel in range(self.nb_outputs):
			std_values_array[num_output_channel]=std_values_file[num_output_channel]




		
		self.mean_fixed.set_value(mean_values_array)
		self.std_fixed.set_value(std_values_array)

	



	def print_parameters(self):
		w=self.W.get_value()
		b=self.b.get_value()
		gamma=self.gamma.get_value()
		beta=self.beta.get_value()
		print("\n\n\n\n\n\n\n\n\n\n\nConvolutional layer "+self.name)
		print("Parameters W (first filter):")

		for input_channel in range(self.w_shape[1]):
			print("Input channel "+str(input_channel))
			print_2D_matrix(w[0][input_channel])

		print("Parameter b (first filter):"+str(b[0]))
		print("Parameter gamma (first filter):"+str(gamma[0]))
		print("Parameter beta (first filter):"+str(beta[0]))







	def save_parameters(self, output_folder_parameters):
		w=self.W.get_value()
		b=self.b.get_value()
		gamma=self.gamma.get_value()
		beta=self.beta.get_value()


		filename_w=output_folder_parameters+"/"+self.name+"_w.npy"
		filename_b=output_folder_parameters+"/"+self.name+"_b.npy"
		filename_gamma=output_folder_parameters+"/"+self.name+"_gamma.npy"
		filename_beta=output_folder_parameters+"/"+self.name+"_beta.npy"


		np.save(filename_w, w)
		np.save(filename_b, b)
		np.save(filename_gamma, gamma)
		np.save(filename_beta, beta)












class Bilinear_Upsampling_Layer2D:
	def __init__(self,str_name, upsampling_ratio, input_layer):
		self.name=str_name
		self.upsampling_ratio=upsampling_ratio
		self.bottom_layer=input_layer.name
		previous_layer_output_tensor_shape=input_layer.output_shape
		self.type="bilinear_upsampling"
		self.output_shape=[previous_layer_output_tensor_shape[0],previous_layer_output_tensor_shape[1],previous_layer_output_tensor_shape[2]*upsampling_ratio,previous_layer_output_tensor_shape[3]*upsampling_ratio]
		self.output=bilinear_upsampling(input_layer.output,upsampling_ratio,batch_size=previous_layer_output_tensor_shape[0],num_input_channels=previous_layer_output_tensor_shape[1])
		#looks correct


		#downsampling/upsampling ratio (1= no downsampling)
		self.factor_y=input_layer.factor_y/self.upsampling_ratio
		self.factor_x=input_layer.factor_x/self.upsampling_ratio



		self.total_offset_y=input_layer.total_offset_y
		self.total_offset_x=input_layer.total_offset_x




class Concatenation_Layer2D:
	def __init__(self,str_name, layer1, layer2):
		#concatenate two layers
		#we assume that the first layer appears before the second one and has bigger dimensions

		#concatenate the cropped version of the first layer and the second layer

		self.name=str_name
		self.type="concatenation"



		str_name_layer1=layer1.name
		str_name_layer2=layer2.name 
		layer1_output_tensor_shape=layer1.output_shape
		layer2_output_tensor_shape=layer2.output_shape
	


		if((layer1.factor_y!=layer2.factor_y) or (layer1.factor_x!=layer2.factor_x)):
			print("PROBLEM WITH DOWNSAMPLING/UPSAMPLING: layer "+self.name)
			sys.exit(3)


		#take into account the fact that in downsampled layers, jump of one line is like a jump of factor_y lines of the input of the network  
		offset_y_concatenation=(layer2.total_offset_y-layer1.total_offset_y)/layer1.factor_y
		offset_x_concatenation=(layer2.total_offset_x-layer1.total_offset_x)/layer1.factor_x


		
		if((offset_y_concatenation<0) or (offset_x_concatenation<0)):
			print("PROBLEM WITH THE CONCATENATION OFFSET: layer "+self.name)
			sys.exit(4)


		self.layer1=str_name_layer1
		self.layer2=str_name_layer2
		self.output_shape=[layer1_output_tensor_shape[0],layer1_output_tensor_shape[1]+layer2_output_tensor_shape[1],layer2_output_tensor_shape[2],layer2_output_tensor_shape[3]]
		self.output=T.concatenate((layer1.output[:,:,offset_y_concatenation:(offset_y_concatenation+layer2_output_tensor_shape[2]),offset_x_concatenation:(offset_x_concatenation+layer2_output_tensor_shape[3])],layer2.output), axis=1)
		#probably correct if provided offset_y_concatenation and offset_x_concatenation are correct
		#print("mon shape:")
		#print(self.output_shape)


		self.factor_y=layer2.factor_y
		self.factor_x=layer2.factor_x



		self.total_offset_y=layer2.total_offset_y
		self.total_offset_x=layer2.total_offset_x












class Concatenation_LayerFC:
	def __init__(self,str_name, layer1, layer2):
		#concatenate two layers
		#we assume that the first layer appears before the second one and has bigger dimensions

		#concatenate the cropped version of the first layer and the second layer

		self.name=str_name
		self.type="concatenation_fc"



		str_name_layer1=layer1.name
		str_name_layer2=layer2.name 
		layer1_output_tensor_shape=layer1.output_shape
		layer2_output_tensor_shape=layer2.output_shape
	


		


		

		


		self.layer1=str_name_layer1
		self.layer2=str_name_layer2

		self.output_shape=[layer1_output_tensor_shape[0],layer1_output_tensor_shape[1]+layer2_output_tensor_shape[1]]
		self.output=T.concatenate((layer1.output,layer2.output), axis=1)
		



	







"""
***********************************************
SOFTMAX AND LOSSES
***********************************************
"""




"""
***
SOFTMAX
***
"""


class SoftmaxLayerClassificationVersion:
	def __init__(self, str_name, input_to_layer, input_tensor_shape):
		"""
		Constructor of the class: define the mathematical expression
		input_layer.output= symbolic Theano matrix of floats (indexes: image, class)
		an object of this class has to contain a mathematical expression of the output of a softmax layer
		"""
		self.output_shape=input_tensor_shape
		self.name=str_name
		self.type="softmax"
		self.output=T.nnet.softmax(input_to_layer)
	




class SoftmaxLayerSegmentationVersion:
	def __init__(self, str_name, input_tensor_shape, input_to_layer):
		"""
		Constructor of the class: define the mathematical expression
		input_layer.output= symbolic Theano 4D tensor of floats (indexes: image, num_class, y, x)
		output=flattened version of the softmax (Theano's softmax wants a 2D tensor...)
		an object of this class has to contain a mathematical expression of the output of a softmax layer
		"""

		self.output_shape=[input_tensor_shape[0]*input_tensor_shape[2]*input_tensor_shape[3],input_tensor_shape[1]]
		

		self.name=str_name
		self.type="softmax"
		#self.output=T.nnet.softmax(matrix_scores)
		self.output=T.nnet.softmax(((input_to_layer.dimshuffle(1,0,2,3)).flatten(2)).dimshuffle(1,0))
		









"""
**********
LOSSES SEGMENTATION
**********
"""



class EntropyLossSegmentationLayer:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		
		#reminder: labels= 3D tensor of integers

		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
	
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)



		
		#self.output=-T.mean(T.log(matrix_softmax_scores)[T.arange(self.vector_labels.shape[0]),self.vector_labels]) + weight_decay*w_norm2_2
		

		
		self.output=-T.mean(T.log(matrix_softmax_scores)[T.arange(self.vector_labels.shape[0]),self.vector_labels])

		




class EntropyThresholdedLossSegmentationLayer:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		
		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
	
		self.output=-T.mean( T.clip(T.log(matrix_softmax_scores)[T.arange(self.vector_labels.shape[0]),self.vector_labels],-2000,-0.01) )
		














class WeightedEntropyLossSegmentationLayer2classes:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (BINARY SEGMENTATION)")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		


		

		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		

		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)

		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[1]-target_proportions[1],target_proportions[1]/nb_examples_class1)

		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
	
		self.weights=self.weights_class0+self.weights_class1

		self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights))
		



class WeightedEntropyLossBinarySegmentationLayer_without_logaritm:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (BINARY SEGMENTATION) WITHOUT LOGARITHM")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		


	


		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		

		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)

		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[1]-target_proportions[1],target_proportions[1]/nb_examples_class1)

		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
	
		self.weights=self.weights_class0+self.weights_class1

		self.output=-(T.sum(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels]*self.weights))
	




class WeightedEntropyLossSegmentationLayer5classes_old:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (5-CLASS SEGMENTATION)")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		



		


		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		


		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)
		examples_class2=T.eq(self.vector_labels,2)
		examples_class3=T.eq(self.vector_labels,3)
		examples_class4=T.eq(self.vector_labels,4)

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)
		nb_examples_class2=T.sum(examples_class2)
		nb_examples_class3=T.sum(examples_class3)
		nb_examples_class4=T.sum(examples_class4)

		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[0]-target_proportions[0],target_proportions[1]/nb_examples_class1)
		self.w_class2=ifelse(T.eq(nb_examples_class2,0),nb_examples_class2/1.0+target_proportions[0]-target_proportions[0],target_proportions[2]/nb_examples_class2)
		self.w_class3=ifelse(T.eq(nb_examples_class3,0),nb_examples_class3/1.0+target_proportions[0]-target_proportions[0],target_proportions[3]/nb_examples_class3)
		self.w_class4=ifelse(T.eq(nb_examples_class4,0),nb_examples_class4/1.0+target_proportions[0]-target_proportions[0],target_proportions[4]/nb_examples_class4)

		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
		self.weights_class2=self.w_class2*T.eq(self.vector_labels,2)
		self.weights_class3=self.w_class3*T.eq(self.vector_labels,3)
		self.weights_class4=self.w_class4*T.eq(self.vector_labels,4)

	
		self.weights=self.weights_class0+self.weights_class1+self.weights_class2+self.weights_class3+self.weights_class4
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)
		

	
		#self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights)/self.sum_weights) + weight_decay*w_norm2_2
		
		self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights))
		





class WeightedEntropyLossSegmentationLayer4classes:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, weights_classes):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (4-CLASS SEGMENTATION)")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		





		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		


		nb_all_examples=self.vector_labels.shape[0]



	
		self.weights_class0=weights_classes[0]*T.eq(self.vector_labels,0)
		self.weights_class1=weights_classes[1]*T.eq(self.vector_labels,1)
		self.weights_class2=weights_classes[2]*T.eq(self.vector_labels,2)
		self.weights_class3=weights_classes[3]*T.eq(self.vector_labels,3)
	
	
		self.weights=self.weights_class0+self.weights_class1+self.weights_class2+self.weights_class3
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)
		


		#self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights)/self.sum_weights) + weight_decay*w_norm2_2
		
		self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights))/10000
		




class WeightedEntropyLossSegmentationLayer4classes2:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		
		#Constructor of the class: define the mathematical expression
		#matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		#tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		##training example can represent an image or a pixel
		#an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (4-CLASS SEGMENTATION) VERSION 2")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		






		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		


		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)
		examples_class2=T.eq(self.vector_labels,2)
		examples_class3=T.eq(self.vector_labels,3)
		

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)
		nb_examples_class2=T.sum(examples_class2)
		nb_examples_class3=T.sum(examples_class3)
		

		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[0]-target_proportions[0],target_proportions[1]/nb_examples_class1)
		self.w_class2=ifelse(T.eq(nb_examples_class2,0),nb_examples_class2/1.0+target_proportions[0]-target_proportions[0],target_proportions[2]/nb_examples_class2)
		self.w_class3=ifelse(T.eq(nb_examples_class3,0),nb_examples_class3/1.0+target_proportions[0]-target_proportions[0],target_proportions[3]/nb_examples_class3)
	
		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
		self.weights_class2=self.w_class2*T.eq(self.vector_labels,2)
		self.weights_class3=self.w_class3*T.eq(self.vector_labels,3)
	
	
		self.weights=self.weights_class0+self.weights_class1+self.weights_class2+self.weights_class3
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)
		


		#self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights)/self.sum_weights) + weight_decay*w_norm2_2
		
		self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights))
		










class WeightedEntropyLossSegmentationLayer3classes:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		
		#Constructor of the class: define the mathematical expression
		#matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		#tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		##training example can represent an image or a pixel
		#an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (3-CLASS SEGMENTATION) VERSION 2")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		





		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		


		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)
		examples_class2=T.eq(self.vector_labels,2)

		

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)
		nb_examples_class2=T.sum(examples_class2)
		

		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[0]-target_proportions[0],target_proportions[1]/nb_examples_class1)
		self.w_class2=ifelse(T.eq(nb_examples_class2,0),nb_examples_class2/1.0+target_proportions[0]-target_proportions[0],target_proportions[2]/nb_examples_class2)
	
		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
		self.weights_class2=self.w_class2*T.eq(self.vector_labels,2)

	
		self.weights=self.weights_class0+self.weights_class1+self.weights_class2
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)
		

		#self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights)/self.sum_weights) + weight_decay*w_norm2_2
		
		self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights))
















class WeightedEntropyLossSegmentationLayer4classes2_no_logarithm:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		
		#Constructor of the class: define the mathematical expression
		#matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		#tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		##training example can represent an image or a pixel
		#an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (4-CLASS SEGMENTATION) VERSION 2 WITHOUT LOGARITHM")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		



	

		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		


		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)
		examples_class2=T.eq(self.vector_labels,2)
		examples_class3=T.eq(self.vector_labels,3)
		

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)
		nb_examples_class2=T.sum(examples_class2)
		nb_examples_class3=T.sum(examples_class3)
		

		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[0]-target_proportions[0],target_proportions[1]/nb_examples_class1)
		self.w_class2=ifelse(T.eq(nb_examples_class2,0),nb_examples_class2/1.0+target_proportions[0]-target_proportions[0],target_proportions[2]/nb_examples_class2)
		self.w_class3=ifelse(T.eq(nb_examples_class3,0),nb_examples_class3/1.0+target_proportions[0]-target_proportions[0],target_proportions[3]/nb_examples_class3)
	
		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
		self.weights_class2=self.w_class2*T.eq(self.vector_labels,2)
		self.weights_class3=self.w_class3*T.eq(self.vector_labels,3)
	
	
		self.weights=self.weights_class0+self.weights_class1+self.weights_class2+self.weights_class3
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)
		

		#self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights)/self.sum_weights) + weight_decay*w_norm2_2
		
		self.output=-(T.sum(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels]*self.weights))
		

	

class WeightedEntropyLossSegmentationLayer4classes_old:
	def __init__(self, str_name, matrix_softmax_scores, shape_scores_before_flattening,offset_y,offset_x, tensor_labels, target_proportions):
		"""
		Constructor of the class: define the mathematical expression
		matrix_softmax_scores= symbolic Theano 2D tensor (indexes: training_exemple, num_class) with the classification scores
		tensor_labels= symbolic Theano 3D vector of integers (indexes= training_exemple, y ,x)
		#training example can represent an image or a pixel
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""

		print("\n\n\n\n\nUSING WEIGHTED SEGMENTATION LOSS (5-CLASS SEGMENTATION)")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy_segmentation"
		




		#crop and flatten the tensor with labels to obtain a vector of integers (index: pixel, value=label)
		self.vector_labels=(tensor_labels[:,offset_y:(offset_y+shape_scores_before_flattening[2]),offset_x:(offset_x+shape_scores_before_flattening[3])]).flatten(1)
		


		examples_class0=T.eq(self.vector_labels,0)
		examples_class1=T.eq(self.vector_labels,1)
		examples_class2=T.eq(self.vector_labels,2)
		examples_class3=T.eq(self.vector_labels,3)

		nb_examples_class0=T.sum(examples_class0)
		nb_examples_class1=T.sum(examples_class1)
		nb_examples_class2=T.sum(examples_class2)
		nb_examples_class3=T.sum(examples_class3)


		nb_all_examples=self.vector_labels.shape[0]



	

		#the 'target_proportions[0]-target_proportions[0]' is just to have float64 instead of int64
		self.w_class0=ifelse(T.eq(nb_examples_class0,0),nb_examples_class0/1.0+target_proportions[0]-target_proportions[0],target_proportions[0]/nb_examples_class0)
		self.w_class1=ifelse(T.eq(nb_examples_class1,0),nb_examples_class1/1.0+target_proportions[0]-target_proportions[0],target_proportions[1]/nb_examples_class1)
		self.w_class2=ifelse(T.eq(nb_examples_class2,0),nb_examples_class2/1.0+target_proportions[0]-target_proportions[0],target_proportions[2]/nb_examples_class2)
		self.w_class3=ifelse(T.eq(nb_examples_class3,0),nb_examples_class3/1.0+target_proportions[0]-target_proportions[0],target_proportions[3]/nb_examples_class3)
		
		self.weights_class0=self.w_class0*T.eq(self.vector_labels,0)
		self.weights_class1=self.w_class1*T.eq(self.vector_labels,1)
		self.weights_class2=self.w_class2*T.eq(self.vector_labels,2)
		self.weights_class3=self.w_class3*T.eq(self.vector_labels,3)
	
	
		self.weights=self.weights_class0+self.weights_class1+self.weights_class2+self.weights_class3
		#vector_labels= symbolic Theano vector of integers (index= training_exemple)
		

		#self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights)/self.sum_weights) + weight_decay*w_norm2_2
		
		self.output=-(T.sum(T.log(matrix_softmax_scores[T.arange(self.vector_labels.shape[0]),self.vector_labels])*self.weights))
		









"""
**********
LOSSES CLASSIFICATION
**********
"""



class EntropyLossLayerClassification:
	def __init__(self, str_name, input_layer, vector_labels):
		"""
		Constructor of the class: define the mathematical expression
		output_layer_scores= symbolic Theano 2D tensor (indexes: image, num_class) with the classification scores
		vector_labels= symbolic Theano vector of integers (index= image)
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""
		print("\n\n\n\n\nCLASSIFICATION: USING STANDARD CLASSIFICATION LOSS")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy"
		#print("nb images="+str(vector_labels.shape[0]))

	
		
		
		#self.output=-T.mean(T.log(output_layer_scores)[T.arange(vector_labels.shape[0]),vector_labels]) + weight_decay*w_norm2_2
		
		self.output=-T.mean(T.log(input_layer.output)[T.arange(vector_labels.shape[0]),vector_labels])
	




class EntropyThresholdedLossLayer:
	def __init__(self, str_name, output_layer_scores, vector_labels):
		"""
		Constructor of the class: define the mathematical expression
		output_layer_scores= symbolic Theano 2D tensor (indexes: image, num_class) with the classification scores
		vector_labels= symbolic Theano vector of integers (index= image)
		an object of this class has to contain a mathematical expression of the output of the entropy loss layer (mean accros the batch)
		"""
		print("\n\n\n\n\nCLASSIFICATION: USING THRESHOLDED CLASSIFICATION LOSS")
		self.output_shape=[1]
		self.name=str_name
		self.type="loss_entropy"
		#print("nb images="+str(vector_labels.shape[0]))

	
		#self.argmax_for_each_image=T.argmax(output_layer_scores,axis=1)
		#self.output=-T.mean( T.clip(T.log(output_layer_scores)[T.arange(vector_labels.shape[0]),vector_labels],-2000,-0.01)) + weight_decay*w_norm2_2
		
		self.output=-T.mean( T.clip(T.log(output_layer_scores)[T.arange(vector_labels.shape[0]),vector_labels],-2000,-0.01))
		














"""
**************************************************************
MULTISTREAM AND MULTILOSS
**************************************************************
"""







class NeuralNet2D(object):

	#def __init__(self, filename_protobuf, input_tensor_shape, str_name, momentum=0.9,learning_rate=0.01,weight_decay=0.0,weights_classes=None,  binary_classification=True, folder_parameters_for_mean_and_std=None, use_alternative_loss_function=False,pad_zeros_input=False, use_loss_function_without_logarithm=False, apply_bn_conv_layers=True,target_shape_input_image=[240,240]):
	def __init__(self, filename_protobuf, input_tensor_shape, str_name, nb_classes, nb_images_gt,test_phase=False,momentum=0.9,learning_rate=0.1,weight_decay=0.0, weight_segmentation=0.7, joint_training=False,  target_shape_input_image=[240,240],weights_classes=None, folder_parameters_for_mean_and_std=None, pad_zeros_input=False, apply_bn_conv_layers=True):
	

	

		#goal: construct the mathematical expressions correspoding to the output, the loss function and the gradients (wrt to the output and wrt to the loss)

		#read the file with the definition of the net and contruct the network
		
		self.list_layers_definition=analyze_network(filename_protobuf)

		name_last_conv_layer=find_name_last_conv_layer(self.list_layers_definition)
		
		self.tensor_labels_0=T.itensor3('tensor_labels')




		self.nb_images_gt=nb_images_gt


		self.list_vectors_labels=[]


		#reminder: one vector of labels per tumor subclass (vectors indexed by num_image)
		for cl in range(1,nb_classes):
			self.list_vectors_labels.append(T.ivector('vector_all_labels'+str(cl)))
		

		self.input_network_0=T.tensor4('input_network')
		self.name=str_name
		#self.nb_images_gt=nb_images_gt


		num_loss_segmentation=0

		num_loss_classification=0


	

		self.momentum=momentum
		#self.momentum=0.0

		#self.learning_rate=0.001
		self.learning_rate=learning_rate
		#self.learning_rate=0.01

		#self.weight_decay=0.004
		self.weight_decay=weight_decay
		#self.weight_decay=0.00001


		if((target_shape_input_image[0]==input_tensor_shape[2]) and (target_shape_input_image[1]==input_tensor_shape[3])):
			pad_zeros_input=False


		#compute the number of zeros to add (4 numbers: bottom, top, left, right)
		#we assume that the targer shapes are larger than the original shape (add zeros)
		
		if(not(pad_zeros_input)):
			self.input_network_padded=self.input_network_0
			self.tensor_labels_padded=self.tensor_labels_0
			self.nb_zeros_top=0
			self.nb_zeros_left=0
		else:

			diff_shape_y=target_shape_input_image[0]-input_tensor_shape[2]

			self.nb_zeros_top=diff_shape_y/2
			nb_zeros_bottom=diff_shape_y-self.nb_zeros_top

			diff_shape_x=target_shape_input_image[1]-input_tensor_shape[3]

			self.nb_zeros_left=diff_shape_x/2
			nb_zeros_right=diff_shape_x-self.nb_zeros_left


			#tensors with zeros
			zeros_top=T.zeros([input_tensor_shape[0],input_tensor_shape[1],self.nb_zeros_top,input_tensor_shape[3]])
			zeros_bottom=T.zeros([input_tensor_shape[0],input_tensor_shape[1],nb_zeros_bottom,input_tensor_shape[3]])

			zeros_left=T.zeros([input_tensor_shape[0],input_tensor_shape[1],target_shape_input_image[0],self.nb_zeros_left])
			zeros_right=T.zeros([input_tensor_shape[0],input_tensor_shape[1],target_shape_input_image[0],nb_zeros_right])

		
			padded_input=T.concatenate((zeros_top,self.input_network_0), axis=2)
			padded_input=T.concatenate((padded_input,zeros_bottom), axis=2)

			padded_input=T.concatenate((zeros_left,padded_input), axis=3)
			padded_input=T.concatenate((padded_input,zeros_right), axis=3)


			self.input_network_padded=padded_input


		



			zeros_top_labels=T.zeros([nb_images_gt,self.nb_zeros_top,input_tensor_shape[3]],dtype=np.int32)
			zeros_bottom_labels=T.zeros([nb_images_gt,nb_zeros_bottom,input_tensor_shape[3]],dtype=np.int32)

			zeros_left_labels=T.zeros([nb_images_gt,target_shape_input_image[0],self.nb_zeros_left],dtype=np.int32)
			zeros_right_labels=T.zeros([nb_images_gt,target_shape_input_image[0],nb_zeros_right],dtype=np.int32)



			padded_labels=T.concatenate([zeros_top_labels,self.tensor_labels_0], axis=1)
			padded_labels=T.concatenate([padded_labels,zeros_bottom_labels], axis=1)

			padded_labels=T.concatenate([zeros_left_labels,padded_labels], axis=2)
			padded_labels=T.concatenate([padded_labels,zeros_right_labels], axis=2)

			self.tensor_labels_padded=padded_labels


			input_tensor_shape=[input_tensor_shape[0],input_tensor_shape[1],target_shape_input_image[0],target_shape_input_image[1]]
			



		



		
		print("shape (modified) of the input")
		print(input_tensor_shape)
		expr_output_previous_layer=self.input_network_padded
		previous_layer_output_tensor_shape=input_tensor_shape


		tensor_input_network=self.input_network_padded
		shape_input_network=input_tensor_shape

		self.parameters=[]
		a=None

		self.list_layers=[]
		self.list_functions_compute_output=[]

		self.w_norm2_2=0.0

		#offsets induced by convolutions and poolings
		#total_offset_y=0
		#total_offset_x=0


		self.nb_weights=0

	

		sum_weights_losses_segmentation=0.0









		k=-1
		for layer_definition in self.list_layers_definition:
			k=k+1


			if(layer_definition.type=="data"):
				#this layer retrieves some channels (precised in the definition file) from the input of the network
				
				a=Data_Layer2D(layer_definition.name, layer_definition.channel_start,layer_definition.channel_end, shape_input_network, tensor_input_network)
				self.list_layers.append(a)






			if(layer_definition.type=="fc"):





				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)



			

				a=FC_Layer2D(layer_definition.name,layer_definition.nb_outputs ,layer_definition.activation, input_layer)
				self.list_layers.append(a)
				self.parameters=self.parameters+a.parameters

				nb_weights_this_layer=a.w_shape[0]*a.w_shape[1]
				self.w_norm2_2=self.w_norm2_2*(self.nb_weights/(self.nb_weights+nb_weights_this_layer))+a.w_norm2_2*(nb_weights_this_layer/(self.nb_weights+nb_weights_this_layer))
				self.nb_weights=self.nb_weights+nb_weights_this_layer



			if(layer_definition.type=="conv"):
 				#find the input layer (to have its output expression, output shape, offsets, upsampling ratios)


				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)
				#print("conv, shape du layer precedent:")
				#print(input_layer.output_shape)




				#specify the number of images to take into account: if it's a segmentation layer, take only the images with the GT
				nb_images_compute_output=-1






				if("segm" in layer_definition.name and not(test_phase)):
					nb_images_compute_output=nb_images_gt
			




				if(apply_bn_conv_layers):
					#apply_bn=(layer_definition.name!=name_last_conv_layer)
					apply_bn=not("segm" in layer_definition.name)
				else:
					apply_bn=False

				if(folder_parameters_for_mean_and_std==None):
					a=ConvLayer2D_bn(layer_definition.name, layer_definition.kernel_size , [layer_definition.stride,layer_definition.stride] ,layer_definition.nb_outputs, layer_definition.activation, input_layer,apply_bn=apply_bn,nb_images_compute_output=nb_images_compute_output)
				else:
					filename_mean_and_std_layer=folder_parameters_for_mean_and_std+"/"+layer_definition.name+"_mean_and_std.txt"
					a=ConvLayer2D_bn(layer_definition.name, layer_definition.kernel_size , [layer_definition.stride,layer_definition.stride] ,layer_definition.nb_outputs, layer_definition.activation, input_layer, filename_mean_and_std_layer,apply_bn=apply_bn,nb_images_compute_output=nb_images_compute_output)
				
				self.list_layers.append(a)
				self.parameters=self.parameters+a.parameters

				
				
				nb_weights_this_layer=a.w_shape[0]*a.w_shape[1]*a.w_shape[2]*a.w_shape[3]
				self.w_norm2_2=self.w_norm2_2*(self.nb_weights/(self.nb_weights+nb_weights_this_layer))+a.w_norm2_2*(nb_weights_this_layer/(self.nb_weights+nb_weights_this_layer))
				self.nb_weights=self.nb_weights+nb_weights_this_layer





				
				

			if(layer_definition.type=="max_pooling"):

				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)

				a=Max_Pooling_Layer2D(layer_definition.name, layer_definition.kernel_size , [layer_definition.stride,layer_definition.stride],  input_layer)
				self.list_layers.append(a)



			if(layer_definition.type=="mean_pooling"):

				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)

				print("\n\n\n\nMEAN POOLING MAN\n\n\n\n\n")
				a=Mean_Pooling_Layer2D(layer_definition.name, layer_definition.kernel_size , [layer_definition.stride,layer_definition.stride],  input_layer)
				self.list_layers.append(a)
				

				


			if(layer_definition.type=="bilinear_upsampling"):
				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)
				#for the classification part
				output_layer_before_upsampling=expr_output_previous_layer
				shape_output_layer_before_upsampling=previous_layer_output_tensor_shape

			


				a=Bilinear_Upsampling_Layer2D(layer_definition.name, layer_definition.upsampling_ratio ,  input_layer)
				self.list_layers.append(a)
			








			if(layer_definition.type=="loss_segmentation"):


				#find the bottom layer
				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)

			

				#check if there is no problem with downsampling/upsampling
				if((input_layer.factor_y!=1) or (input_layer.factor_x!=1)):
					print("ERROR WITH DOWNSAMPLING/UPSAMPLING (architecture of the network): layer "+input_layer.name+", factor_y="+str(input_layer.factor_y)+",factor_x="+str(input_layer.factor_x))
					sys.exit(2)




				#take the softmax

				



				output_segmentation=input_layer.output
				shape_segmentation=input_layer.output_shape		


				softmax_this_layer=SoftmaxLayerSegmentationVersion('softmax',shape_segmentation,output_segmentation)








				#take the loss
				if(weights_classes!=None):

					if(len(weights_classes)!=nb_classes):
						print("Problem with the number of classes and the loss function")
						sys.exit(1)

					if(nb_classes==4):
						loss_layer=WeightedEntropyLossSegmentationLayer4classes2('loss_s'+str(num_loss_segmentation),softmax_this_layer.output,input_layer.output_shape, input_layer.total_offset_y,input_layer.total_offset_x,self.tensor_labels_padded,weights_classes)
					elif(nb_classes==3):
						loss_layer=WeightedEntropyLossSegmentationLayer3classes('loss_s'+str(num_loss_segmentation),softmax_this_layer.output,input_layer.output_shape, input_layer.total_offset_y,input_layer.total_offset_x,self.tensor_labels_padded,weights_classes)

					elif(nb_classes==2):
						loss_layer=WeightedEntropyLossSegmentationLayer2classes('loss_s'+str(num_loss_segmentation),softmax_this_layer.output,input_layer.output_shape, input_layer.total_offset_y,input_layer.total_offset_x,self.tensor_labels_padded,weights_classes)


					else:
						print("Problem with the number of classes and the loss function")
						sys.exit(1)

				else:
					loss_layer=EntropyLossSegmentationLayer('loss_s'+str(num_loss_segmentation),softmax_this_layer.output,input_layer.output_shape, input_layer.total_offset_y,input_layer.total_offset_x,self.tensor_labels_padded)
					




				sum_weights_losses_segmentation=sum_weights_losses_segmentation+layer_definition.weight_loss
				
				if(num_loss_segmentation==0):
					total_loss_segmentation=loss_layer.output*layer_definition.weight_loss
				else:
					total_loss_segmentation=total_loss_segmentation+loss_layer.output*layer_definition.weight_loss



				self.list_layers.append(loss_layer)
				num_loss_segmentation=num_loss_segmentation+1






			if(layer_definition.type=="loss_classification"):






			
				if(not(joint_training)):
					continue






				#find the bottom layer
				input_layer=find_layer(layer_definition.bottom_layer,self.list_layers)

				softmax_layer_classification=SoftmaxLayerClassificationVersion('softmax',input_layer.output,input_layer.output_shape)
				self.list_layers.append(softmax_layer_classification)

			

				class_loss=layer_definition.class_loss
				index_class=class_loss-1
				classification_loss_layer=EntropyLossLayerClassification('loss_c'+str(class_loss),softmax_layer_classification, self.list_vectors_labels[index_class])



		

				if(num_loss_classification==0):
					total_loss_classification=classification_loss_layer.output
				else:
					total_loss_classification=total_loss_classification+classification_loss_layer.output


				num_loss_classification=num_loss_classification+1

			


			if(layer_definition.type=="concatenation"):
				#find the two layers and the offsets between them taking into account the downsampling and upsamplings between the two layers
				layer1=find_layer(layer_definition.layer1,self.list_layers)
				layer2=find_layer(layer_definition.layer2,self.list_layers)
				

				a=Concatenation_Layer2D(layer_definition.name, layer1,layer2)
				self.list_layers.append(a)

			

			if(layer_definition.type=="concatenation_fc"):
				#find the two layers and the offsets between them taking into account the downsampling and upsamplings between the two layers
				layer1=find_layer(layer_definition.layer1,self.list_layers)
				layer2=find_layer(layer_definition.layer2,self.list_layers)
				

				a=Concatenation_LayerFC(layer_definition.name, layer1,layer2)
				self.list_layers.append(a)


		#SEGMENTATION LOSS

		#find the last convolutional layer
		last_conv_layer=find_layer(name_last_conv_layer,self.list_layers)


		print("last_conv_layer: "+last_conv_layer.name)




	

		self.output_segmentation_scores=last_conv_layer.output

		self.total_offset_x=last_conv_layer.total_offset_x
		self.total_offset_y=last_conv_layer.total_offset_y



		

		print("\n\n\nsum_weights_losses_segmentation="+str(sum_weights_losses_segmentation))

		#FINAL LOSS



		if(joint_training and (num_loss_classification!=(nb_classes-1))):
			print("Problem with the classification loss")
			sys.exit(1)

		




		if(not(test_phase)):

			if(joint_training and (sum_weights_losses_segmentation>0.0)):
				total_loss_classification=total_loss_classification/(float(nb_classes-1))
				self.loss= weight_segmentation*total_loss_segmentation+(1.0- weight_segmentation)*total_loss_classification

				self.segmentation_loss=total_loss_segmentation
				self.classification_loss=total_loss_classification
			elif(sum_weights_losses_segmentation==0.0):	
				print("\n\n\nNO SEGMENTATION LOSS")
				self.loss= total_loss_classification

			else:
				self.loss= total_loss_segmentation



			
			self.gradients_loss_wrt_parameters=[T.grad(self.loss, parameter) for parameter in self.parameters]



		


		if(joint_training):
			#if(nb_classes==3):
			self.accuracy=T.sum(T.eq(T.argmax(softmax_layer_classification.output,axis=1),self.list_vectors_labels[0]))*(1.0/input_tensor_shape[0])


	

	def print_layers(self):
		print("\n\n\n\n\n\n Layers of the net:")
		for layer in self.list_layers:
			print("\nlayer name: "+layer.name)
			#print("\nbottom layer: "+layer.bottom_layer)
			
			if(layer.type=="conv" or layer.type=="fc"):
				print("\nbottom layer: "+layer.bottom_layer)
				print("activation: "+layer.activation)

			if(layer.type=="bilinear_upsampling"):
				print("\nbottom layer: "+layer.bottom_layer)

			#if(layer.type=="loss_segmentation"):
			#	print("\nbottom layer: "+layer.bottom_layer)
			if(layer.type=="concatenation"):
				print("concat_layer1: "+layer.layer1)
				print("concat_layer2: "+layer.layer2)

			if(layer.type=="conv"):
				print("Filter shape: ["+",".join([str(k) for k in layer.w_shape])+"]")
				print("stride: ["+",".join([str(k) for k in layer.strides])+"]")

			if(layer.type=="max_pooling"):
				print("\nbottom layer: "+layer.bottom_layer)
				print("Kernel shape: ["+",".join([str(k) for k in layer.kernel_shape])+"]")
				print("stride: ["+",".join([str(k) for k in layer.strides])+"]")
				

			if(layer.type=="conv" or layer.type=="fc"):
				print("Number of outputs: "+str(layer.nb_outputs))
			

			if((layer.type=="conv") or (layer.type=="max_pooling")):
				print("total_offset_y: "+str(layer.total_offset_y))
				print("total_offset_x: "+str(layer.total_offset_x))
				print("factor_y: "+str(layer.factor_y))
				print("factor_x: "+str(layer.factor_x))

			print("layer type: "+layer.type)
			print("layer output dimensions: ["+",".join([str(k) for k in layer.output_shape])+"]")





	


	def set_parameters_all_layers(self, folder_parameters):
		for layer in self.list_layers:
			if(layer.type=="conv"):
				layer.set_parameters(folder_parameters)



			if(layer.type=="fc"):
				layer.set_parameters(folder_parameters)




	def set_parameters_few_layers(self, folder_parameters):
		for layer in self.list_layers:
			if((layer.type=="conv") or (layer.type=="fc")):
				if(os.path.isfile(folder_parameters+"/"+layer.name+"_w.npy")):
					layer.set_parameters(folder_parameters)
				else:
					print("couldn't find the parameters for the layer "+layer.name)




	def save_parameters_all_layers(self, output_folder_parameters):
		#os.system("mkdir "+output_folder_parameters)
		recursive_mkdir(output_folder_parameters)
		for layer in self.list_layers:
			if((layer.type=="conv")):
				layer.save_parameters(output_folder_parameters)


			if((layer.type=="fc")):
				layer.save_parameters(output_folder_parameters)