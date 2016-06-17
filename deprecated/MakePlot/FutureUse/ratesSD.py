import csv
import numpy as np
import glob
import os

def findave(name1, name2, name3, levels): #levels is total number of degrade levels 
	N = len(glob.glob('stat/rates/*'))/6 #change root, divide by 2 if necessary 
	f_all = np.zeros(shape=(6,0)) 

	for num in range(0,N):
         a = np.genfromtxt(name1+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine1 = np.column_stack((f_all[:,1],f_all[:,8]))
	
	for num2 in range(0,N-2):
         f_combine1 = np.column_stack((f_combine1,f_all[:,15+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name2+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine2 = np.column_stack((f_all[:,1],f_all[:,8]))
	
	for num2 in range(0,N-2):
         f_combine2 = np.column_stack((f_combine2,f_all[:,15+7*num2]))


########

	for num in range(0,N):
         a = np.genfromtxt(name3+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine3 = np.column_stack((f_all[:,1],f_all[:,8]))
	
	for num2 in range(0,N-2):
         f_combine3 = np.column_stack((f_combine3,f_all[:,15+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name1+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine4 = np.column_stack((f_all[:,2],f_all[:,9]))
	
	for num2 in range(0,N-2):
         f_combine4 = np.column_stack((f_combine4,f_all[:,16+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name2+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine5 = np.column_stack((f_all[:,2],f_all[:,9]))
	
	for num2 in range(0,N-2):
         f_combine5 = np.column_stack((f_combine5,f_all[:,16+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name3+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine6 = np.column_stack((f_all[:,2],f_all[:,9]))
	
	for num2 in range(0,N-2):
         f_combine6 = np.column_stack((f_combine6,f_all[:,16+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name1+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine7 = np.column_stack((f_all[:,3],f_all[:,10]))
	
	for num2 in range(0,N-2):
         f_combine7 = np.column_stack((f_combine7,f_all[:,17+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name2+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine8 = np.column_stack((f_all[:,3],f_all[:,10]))
	
	for num2 in range(0,N-2):
         f_combine8 = np.column_stack((f_combine8,f_all[:,17+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name3+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine9 = np.column_stack((f_all[:,3],f_all[:,10]))
	
	for num2 in range(0,N-2):
         f_combine9 = np.column_stack((f_combine9,f_all[:,17+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name1+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine10 = np.column_stack((f_all[:,4],f_all[:,11]))
	
	for num2 in range(0,N-2):
         f_combine10 = np.column_stack((f_combine10,f_all[:,18+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name2+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine11 = np.column_stack((f_all[:,4],f_all[:,11]))
	
	for num2 in range(0,N-2):
         f_combine11 = np.column_stack((f_combine11,f_all[:,18+7*num2]))

########

	for num in range(0,N):
         a = np.genfromtxt(name3+str(num)+".csv",skip_header=1,delimiter="\t") #change root
         f_all = np.hstack((f_all,a)) 

	f_combine12 = np.column_stack((f_all[:,4],f_all[:,11]))
	
	for num2 in range(0,N-2):
         f_combine12 = np.column_stack((f_combine12,f_all[:,18+7*num2]))


######


	sd1 = np.std(f_combine1, axis=1, dtype=np.float64) #base, TP
	sd2 = np.std(f_combine2, axis=1, dtype=np.float64) #deg, TP
	sd3 = np.std(f_combine3, axis=1, dtype=np.float64) #noisy, TP
	sd4 = np.std(f_combine4, axis=1, dtype=np.float64) #base, TN
	sd5 = np.std(f_combine5, axis=1, dtype=np.float64) #deg, TN
	sd6 = np.std(f_combine6, axis=1, dtype=np.float64)	#noisy, TN
	sd7 = np.std(f_combine7, axis=1, dtype=np.float64)
	sd8 = np.std(f_combine8, axis=1, dtype=np.float64)
	sd9 = np.std(f_combine9, axis=1, dtype=np.float64)
	sd10 = np.std(f_combine10, axis=1, dtype=np.float64)
	sd11 = np.std(f_combine11, axis=1, dtype=np.float64)
	sd12 = np.std(f_combine12, axis=1, dtype=np.float64)


	sd = np.column_stack((sd1, sd2, sd3, sd4, sd5, sd6, sd7, sd8, sd9, sd10, sd11, sd12))

	np.savetxt("SD_output_rates_gaussian.csv", sd, delimiter=",")


findave("stat/rates/rates_base_['gaussian']_","stat/rates/rates_deg_['gaussian']_","stat/rates/rates_noisy_['gaussian']_",6)