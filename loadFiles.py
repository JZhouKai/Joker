'''
script to load TIMIT data into one file 
switch which function to use at end of file
'''

from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility
import glob
from scipy.io import wavfile
from scipy.signal import decimate


#----------------------- define trimming function --------------------------# 	

def loadTrim():

	# get all names of files in database
	fileNamesTrain = np.array([])
	os.chdir(r"F:\Eclipes\timit\timit\zk_train")
	for file in glob.glob("*.wav"):
                fileNamesTrain = np.append(fileNamesTrain, file)

	# define how many files to use
	slicedFileNamesTrain = fileNamesTrain[0:2000]
	
	# get random order to spread the speakers
	np.random.shuffle(slicedFileNamesTrain)

	trimmed_output_train = np.array([])
	for fileName in slicedFileNamesTrain:
		Fs, newFile = wavfile.read(fileName)

		# replace "silence" with 0
		newFile[abs(newFile) < 10] = 0	
		
		# remove zeros at beginning and end of file	
		trimmedFile = np.trim_zeros(newFile)
		trimmed_output_train = np.append(trimmed_output_train, trimmedFile)

	# define output as int16 in order to avoid clipping using wavfile.write
	int_array = trimmed_output_train.astype("int16")
	
	wavfile.write('7_21_2000originalTrainingFiles.wav', 16000, int_array)
	print ("file written")

#------------------- define function to remove "silence" -----------------# 	

def loadNoZeros():

	# get all names of files in database
	fileNamesTrain = np.array([])
	os.chdir(r"F:\Eclipes\timit\timit\zk_train")
	for file in glob.glob('*.wav'):
                fileNamesTrain = np.append(fileNamesTrain, file)
                
	# define how many files to use
	slicedFileNamesTrain = fileNamesTrain[0:2000]
	
	# get random order to spread the speakers
	np.random.shuffle(slicedFileNamesTrain)
	
	output_train = np.array([])
	for fileName in slicedFileNamesTrain:
		Fs, newFile = wavfile.read(fileName)
		
		# only store the values greater than 10
		noZeroes = newFile[abs(newFile) > 10]
		output_train = np.append(output_train, noZeroes)

	# define output as int16 in order to avoid clipping using wavfile.write
	int_array = output_train.astype("int16")
	
	wavfile.write('noZeros.wav', 16000, int_array)
	print ("file written")



#--------------- define function to downsample signal ------------------# 

def downsample():

	Fs, newFile = wavfile.read("trimmedTestFB_10.wav")
	file = newFile[0:1000000]
	
	downsampled = decimate(file, 2, n=61, ftype="fir")
	
	wavfile.write('downsampledTest8000.wav', 8000, downsampled)
	print("file written")
	
#---------------------- run functions ----------------------# 

loadTrim()
#loadNoZeros()
#downsample()

