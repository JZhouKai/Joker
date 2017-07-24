'''
functions to apply on data before they are used in the neural network 
the function "hanningFFT" is the one being called from main.py 
'''

import glob
import numpy as np
from scipy.io import wavfile
import math
from scipy.signal import lfilter
from numpy.random import randn
from numpy.fft import fft
import random

def applyHanning(inputSignal, N, windowLength):

	print("sample length before cut off in 'applyHanning': ", len(inputSignal))
	
	newN = N-N%windowLength
	numberOfWindows = math.floor(newN/windowLength)*2
	
	# cut inputSignal to be a multiple of windowLength < N
	inputSignal = inputSignal[0:newN]
	print("sample length after cut off in 'appltHanning': ", len(inputSignal))
	print(inputSignal)
	# make hanning window
	window = np.hanning(windowLength)

	# iterate through the whole input signal and copy windowed batches into hanningArray
	hanningArray = np.zeros(shape=(numberOfWindows,windowLength))
	j = 0;
	index = 0
	while (index < newN-windowLength):
		hanningArray[j] = inputSignal[index:int(index+windowLength)]*window
		index += windowLength/2
		j += 1
	
	return hanningArray

def doFFT(hanningArray):

	# take fft of each row 
	fftArray = np.apply_along_axis(np.fft.fft, axis=1, arr=hanningArray)

	# storing phase (in radians) from output of fft
	storedPhase = np.apply_along_axis(np.angle, axis=1, arr=fftArray) 

	# extracting amplitude from output of fft
	amplitude = np.apply_along_axis(np.absolute, axis=1, arr = fftArray)

	return [amplitude, storedPhase]

# stacking signal so that the neural network "sees" more than the one being trained
def stack(inputMatrix, windowLength):
	# get shape
	rows = inputMatrix.shape[0]	
	columns = inputMatrix.shape[1]
	
	# should stack 5 rows for the one in the middle. 
	zeroVector = np.zeros(columns)
	L = windowLength/2 + 1
	stackedMatrix = np.zeros(shape=(rows, columns*5))
		
	stackedMatrix[0][0:L] = zeroVector
	stackedMatrix[0][L:2*L] = zeroVector
	stackedMatrix[0][2*L:3*L] = inputMatrix[0]
	stackedMatrix[0][3*L:4*L] = inputMatrix[1]
	stackedMatrix[0][4*L:5*L] = inputMatrix[2]
	
	stackedMatrix[1][0:L] = zeroVector
	stackedMatrix[1][L:2*L] = inputMatrix[0]
	stackedMatrix[1][2*L:3*L] = inputMatrix[1]
	stackedMatrix[1][3*L:4*L] = inputMatrix[2]
	stackedMatrix[1][4*L:5*L] = inputMatrix[3]
		
	for i in range (0, rows-3):
		stackedMatrix[i][0:L] = inputMatrix[i-2]
		stackedMatrix[i][L:2*L] = inputMatrix[i-1]
		stackedMatrix[i][2*L:3*L] = inputMatrix[i]
		stackedMatrix[i][3*L:4*L] = inputMatrix[i+1]
		stackedMatrix[i][4*L:5*L] = inputMatrix[i+2]
	
	stackedMatrix[rows-2][0:L] = inputMatrix[rows-4]
	stackedMatrix[rows-2][L:2*L] = inputMatrix[rows-3]
	stackedMatrix[rows-2][2*L:3*L] = inputMatrix[rows-2]
	stackedMatrix[rows-2][3*L:4*L] = inputMatrix[rows-1]
	stackedMatrix[rows-2][4*L:5*L] = zeroVector
	
	stackedMatrix[rows-1][0:L] = inputMatrix[rows-3]
	stackedMatrix[rows-1][L:2*L] = inputMatrix[rows-2]
	stackedMatrix[rows-1][2*L:3*L] = inputMatrix[rows-1]
	stackedMatrix[rows-1][3*L:4*L] = zeroVector
	stackedMatrix[rows-1][4*L:5*L] = zeroVector
	
	return stackedMatrix

# the function called from main.py calls the other functions in this script
def hanningFFT(inputMatrix, N, windowLength):
	hanningArray = applyHanning(inputMatrix, N, windowLength)
	amplitude, phase = doFFT(hanningArray)
	halfAmplitude = amplitude[0:, 0:int((windowLength/2)+1)]
	abslog = np.log10(halfAmplitude+1)
	
	# because the input to the keras model should be float32
	abslog = abslog.astype('float32')	
	
	# to get the dB-value defined as 10log(x)
	# return (abslog * 10), phase
	return abslog, phase

	



