import os
import sys

path_folder_input_all_folds="test600"
enable_replacing=False












"""
path_folder_result_all_folds="test601"


nb_WA=50

index_start_WA=0

"""


"""
path_folder_result_all_folds="test602"


nb_WA=50

index_start_WA=50
"""



"""
path_folder_result_all_folds="test603"


nb_WA=50

index_start_WA=100
"""



"""
path_folder_result_all_folds="test610"


nb_WA=100

index_start_WA=0
"""

"""
path_folder_result_all_folds="test620"


nb_WA=100

index_start_WA=50
"""



"""
path_folder_result_all_folds="test630"


nb_WA=100

index_start_WA=100


if(not(enable_replacing) and os.path.isdir(path_folder_result_all_folds)):
	print("already exists")
	sys.exit(1)

"""






"""
path_folder_result_all_folds="test611"


nb_WA=150

index_start_WA=0
"""



path_folder_result_all_folds="test612"


nb_WA=150

index_start_WA=50












if(not(enable_replacing) and os.path.isdir(path_folder_result_all_folds)):
	print("already exists")
	sys.exit(1)

os.system("mkdir -p "+path_folder_result_all_folds)


def choose_patients_WA(path_file_input_WA,nb_WA,index_start_WA):

	lines_input=open(path_file_input_WA,"r").read().split("\n")


	result=[]

	for k in range(nb_WA):

		index=(index_start_WA+k)%len(lines_input)

		result.append(lines_input[index])


	return result




file_config=open(path_folder_result_all_folds+"/config.txt","w")

file_config.write("nb_WA="+str(nb_WA)+"\nindex_start_WA="+str(index_start_WA))

file_config.close()




folders_folds=os.listdir(path_folder_input_all_folds)


for name_fold in folders_folds:


	path_folder_input=path_folder_input_all_folds+"/"+str(name_fold)

	path_folder_result=path_folder_result_all_folds+"/"+str(name_fold)


	os.system("mkdir -p "+path_folder_result)


	files_input=os.listdir(path_folder_input)




	for f in files_input:

		path_file_input=path_folder_input+"/"+f

		path_file_output=path_folder_result+"/"+f


		print(path_file_input)
		print(path_file_output)
		#sys.exit(0)

		if(os.path.isfile(path_file_output)):
			print("already exists")
			sys.exit(1)


		if(not('without_gt' in f)):
			#copy
			os.system("cp "+path_file_input+" "+path_file_output)
			continue

		else:

			file_output=open(path_file_output,"w")

			patients_WA=choose_patients_WA(path_file_input_WA=path_file_input,nb_WA=nb_WA,index_start_WA=index_start_WA)

			file_output.write("\n".join(patients_WA))

			file_output.close()