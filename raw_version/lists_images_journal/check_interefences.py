import os


filename1="training_GT10BRATS_242_W/miccai2_training_patients_flair_axial_with_gt.txt"
filename2="training_GT10BRATS_242_W/miccai2_training_patients_flair_axial_without_gt.txt"
filename3="miccai2_test_patients_flair_axial_with_gt.txt"

lines_1=open(filename1,"r").read().split("\n")
lines_2=open(filename2,"r").read().split("\n")
lines_3=open(filename3,"r").read().split("\n")

print(len(lines_1))
print(len(lines_2))
print(len(lines_3))

for name1 in lines_2:
	for name2 in lines_1:
		if(name1.replace("//","/")==name2.replace("//","/")):
			print(name1+" apparait dans les deux fichiers")


#print(lines_3)

j=0
for f in [filename1, filename2, filename3]:
	j=j+1

	if(j==1):
		lines=lines_1
	if(j==2):
		lines=lines_2
	if(j==3):
		lines=lines_3


	#k=-1
	for k in range(len(lines)):
		name=lines[k]
		pat1=name.split("/")[-1]
		#k=k+1
		a=k
		for name2 in lines[k+1:]:
			pat2=name2.split("/")[-1]
			a=a+1
			#if(name.replace("//","/")==name2.replace("//","/")):
			if(pat1==pat2):
				print("\n\n\nlignes "+str(k+1)+","+str(a+1)+") "+name +" est un doublon dans le fichier "+f)



#find patients which have disappeared
folder_all_patients_hgg="/data/asclepios/user/pmlynars/slices_bin_normalized_without_registration/T2_FLAIR/BRATS2015/train/HGG/axial"
folder_all_patients_lgg="/data/asclepios/user/pmlynars/slices_bin_normalized_without_registration/T2_FLAIR/BRATS2015/train/LGG/axial"

print(len(os.listdir(folder_all_patients_hgg)))
print(len(os.listdir(folder_all_patients_lgg)))



folder_all_patients=folder_all_patients_lgg
all_patients=os.listdir(folder_all_patients)
#print("aaa")
#print(len(lines_1))

#ATTENTION TO '//' IN PATHS

for folder_patient in all_patients:
	path_folder_patient=folder_all_patients+"/"+folder_patient
	patient_found=False
	for path in lines_1:
		if("///" in path):
			print("path bizarre "+path)

		if(not("train" in path)):
			print("path bizarre "+path)
		#print("comparing "+path_folder_patient+" and "+path)
		if(path_folder_patient.replace("//","/")==path.replace("//","/")):
			patient_found=True
			break

	for path in lines_2:
		if(not("train" in path)):
			print("path bizarre "+path)
		if("///" in path):
			print("path bizarre "+path)
		if(patient_found):
			break
		if(path_folder_patient.replace("//","/")==path.replace("//","/")):
			patient_found=True
			break

	for path in lines_3:
		if(not("train" in path)):
			print("path bizarre "+path)
		if("///" in path):
			print("path bizarre "+path)
		if(patient_found):
			break
		if(path_folder_patient.replace("//","/")==path.replace("//","/")):
			patient_found=True
			break

	if(not(patient_found)):
		print(folder_patient+ " was not found")