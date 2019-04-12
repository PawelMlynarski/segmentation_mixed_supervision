import os
import sys


path_folder_one_fold=sys.argv[1]




filenames=os.listdir(path_folder_one_fold)


for filename in filenames:
	path_file=path_folder_one_fold+"/"+filename

	if("axial" in filename):
		print("file "+filename)
		path_output_sagittal=path_file.replace("axial","sagittal")
		path_output_coronal=path_file.replace("axial","coronal")

		content_file_axial=open(path_file,"r").read()


		content_file_sagittal=content_file_axial.replace("axial","sagittal")
		content_file_coronal=content_file_axial.replace("axial","coronal")



		file_sagittal=open(path_output_sagittal,"w")
		file_sagittal.write(content_file_sagittal)
		file_sagittal.close()


		file_coronal=open(path_output_coronal,"w")
		file_coronal.write(content_file_coronal)
		file_coronal.close()