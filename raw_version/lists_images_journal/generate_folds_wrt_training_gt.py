import os
import sys
import random

filename_paths_images="list_all_patients_flair_axial.txt"


#there are 186 images


num_test=int(sys.argv[1])


nb_folds=int(sys.argv[2])

print("Nu of the test: "+str(num_test))
print("Generating "+str(nb_folds)+" folds")

#nb_images_test=40
nb_images_test=50

folder_all_folds="test"+str(num_test)


#read the list of images 

list_paths_flair_patients=open(filename_paths_images,"r").read().split("\n")

nb_total_images=len(list_paths_flair_patients)

nb_images_training_with_gt=nb_total_images/nb_folds

nb_images_training_without_gt=nb_total_images- nb_images_training_with_gt - nb_images_test

if(nb_images_training_with_gt!=(int(round(float(nb_total_images)/float(nb_folds))))):
	print("ATTENTION: nb_total_images is not divisble by nb_images_training_with_gt")
	sys.exit(1)


print("nb_total_images: "+str(nb_total_images))
print("nb_images_training_with_gt: "+str(nb_images_training_with_gt))
print("nb_images_training_without_gt: "+str(nb_images_training_without_gt))
print("nb_images_test: "+str(nb_images_test))


#randomly shuffle the list
random.shuffle(list_paths_flair_patients)




def write_list_into_file(listt, filename_result):
	file_output=open(filename_result,"w")
	file_output.write("\n".join(listt))
	file_output.close()



def check_interferences(list_paths_patients_training_with_gt,list_paths_patients_training_without_gt,list_paths_patients_test):
	print("Checking intererences...")
	for pat_training_with_gt in list_paths_patients_training_with_gt:
		if(pat_training_with_gt in list_paths_patients_training_without_gt):
			print("Interference (training with GT vs training without GT) for the patient "+pat_training_with_gt)
			sys.exit(4)


		if(pat_training_with_gt in list_paths_patients_test):
			print("Interference (training with GT vs patients test) for the patient "+pat_training_with_gt)
			sys.exit(4)

	for pat_test in list_paths_patients_test:
		if(pat_test in list_paths_patients_training_without_gt):
			print("Interference (paitents test vs training without GT) for the patient "+pat_test)
			sys.exit(4)

	print("Test ok")



def create_lists_one_fold(list_paths_flair_patients, index_images_training_with_gt, nb_images_training_with_gt,nb_images_training_without_gt,nb_images_test):
	result_list_training_with_gt=list_paths_flair_patients[index_images_training_with_gt:(index_images_training_with_gt+nb_images_training_with_gt)]
	
	nb_total_images=len(list_paths_flair_patients)

	result_list_training_without_gt=[]
	result_list_test=[]



	for num_test_image in range(nb_images_test):
		index=(index_images_training_with_gt+nb_images_training_with_gt+num_test_image)%nb_total_images
		result_list_test.append(list_paths_flair_patients[index])


	for path_patient in list_paths_flair_patients:
		add_patient=True
		for pat_training_with_gt in result_list_training_with_gt:
			if(path_patient==pat_training_with_gt):
				add_patient=False
				break

		if(add_patient):
			for pat_test in result_list_test:
				if(path_patient==pat_test):
					add_patient=False
					break


		if(add_patient):
			result_list_training_without_gt.append(path_patient)



	return (result_list_training_with_gt,result_list_training_without_gt,result_list_test)
		


#create k folds

index_images_training_with_gt=0

for num_fold in range(nb_folds):
	index_images_training_with_gt=num_fold*nb_images_training_with_gt
	name_folder_one_fold=folder_all_folds+"/fold"+str(num_fold+1)
	os.system("mkdir -p "+name_folder_one_fold)

	#create 3 files: patients training with GT, patients training without GT, patients test
	list_paths_patients_training_with_gt,list_paths_patients_training_without_gt,list_paths_patients_test=create_lists_one_fold(list_paths_flair_patients, index_images_training_with_gt, nb_images_training_with_gt,nb_images_training_without_gt,nb_images_test)


	if(len(list_paths_patients_training_with_gt)!=nb_images_training_with_gt):
		print("Some problem with the number of training images with GT ")
		print("expected:"+str(nb_images_training_with_gt))
		print("found:"+str(len(list_paths_patients_training_with_gt)))
		sys.exit(2)

	if(len(list_paths_patients_training_without_gt)!=nb_images_training_without_gt):
		print("Some problem with the number of training images without GT")
		print("expected:"+str(nb_images_training_without_gt))
		print("found:"+str(len(list_paths_patients_training_without_gt)))
		sys.exit(2)

	if(len(list_paths_patients_test)!=nb_images_test):
		print("Some problem with the number of test images")
		sys.exit(2)

	filename_patients_training_with_gt_this_fold=name_folder_one_fold+"/training_patients_flair_axial_with_gt.txt"
	filename_patients_training_without_gt_this_fold=name_folder_one_fold+"/training_patients_flair_axial_without_gt.txt"
	filename_patients_test_this_fold=name_folder_one_fold+"/test_patients_flair_axial_with_gt.txt"


	


	check_interferences(list_paths_patients_training_with_gt,list_paths_patients_training_without_gt,list_paths_patients_test)

	write_list_into_file(list_paths_patients_training_with_gt,filename_patients_training_with_gt_this_fold)
	write_list_into_file(list_paths_patients_training_without_gt,filename_patients_training_without_gt_this_fold)
	write_list_into_file(list_paths_patients_test,filename_patients_test_this_fold)



	#fold with all the training images (only for standard training)

	num_fold_all_training_images=num_fold+10
	name_folder_one_fold_all_training_images=name_folder_one_fold.replace("fold"+str(num_fold+1),"fold"+str(num_fold_all_training_images+1))
	print("creating folder "+name_folder_one_fold_all_training_images)
	os.system("mkdir -p "+name_folder_one_fold_all_training_images)

	filename_patients_training_with_gt_this_fold_all_patients=name_folder_one_fold_all_training_images+"/training_patients_flair_axial_with_gt.txt"
	filename_patients_test_this_fold_all_patients=name_folder_one_fold_all_training_images+"/test_patients_flair_axial_with_gt.txt"
	



	write_list_into_file(list_paths_patients_training_with_gt+list_paths_patients_training_without_gt,filename_patients_training_with_gt_this_fold_all_patients)
	write_list_into_file(list_paths_patients_test,filename_patients_test_this_fold_all_patients)