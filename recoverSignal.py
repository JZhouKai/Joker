

'''
Script to recover speech signal from output of the keras model
The output is the FFT-abslog of the original signal.
Should "mirror" the signal before taking the IFFT 
''' 


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import glob
from scipy.io import wavfile
from scipy.signal import decimate


def do_ifft(abslog, storedPhase): 

	numberOfWindows = abslog.shape[0]
	print("numberOfWindows: ", numberOfWindows)
	windowLength = (abslog.shape[1]-1)*2
	print("windowLength: ", windowLength)

	# get the linear signal
	amplitude = np.power(10,abslog)

	# reversing the amplitude array in order to create mirror 
	reversedArray = amplitude[:, ::-1]

	# the stored values are more than half of the original signal
	reversedArray = reversedArray[:, 1:windowLength/2]

	# in order to do the ifft, we must reconstruct the other half of the fft signal
	fullAmplitude = np.concatenate((amplitude, reversedArray), axis=1)

	# need to put together the amplitude and phase, z = Acos(0)+jAsin(0)  (0 is angle)
	angleAndAmplitude = np.multiply(fullAmplitude, np.exp(storedPhase*1j))

	# restoring the original matrix
	ifftArray = np.apply_along_axis(np.fft.ifft, axis=1, arr=angleAndAmplitude)

	# take out the real part of the ifft
	realOutputMatrix = ifftArray.real

	# need to do overlap-add.
	# ifftArray is now a 256x2524 matrix. should be overlapped into one 323301 (323200) array
	overlap = np.zeros(numberOfWindows*windowLength/2+windowLength/2)

	overlap[0:windowLength/2] = realOutputMatrix[0][0:windowLength/2]
	j=windowLength/2 	#current index in overlap array
	for i in range(0,int(numberOfWindows-1)):
		overlap[j:j+windowLength/2] = np.add(realOutputMatrix[i][windowLength/2:],realOutputMatrix[i+1][0:windowLength/2])
		j += windowLength/2

	# must add the last values
	overlap[j:] = realOutputMatrix[numberOfWindows-1][windowLength/2:]

	return overlap
	
	
def mirrorSignal(signal, windowLength):
	
	# reversing the amplitude array in order to create mirror 
	reversedArray = signal[:, ::-1]
	
	# the stored values are more than half of the original signal
	reversedArray = reversedArray[:, 1:int(windowLength/2)]

	# in order to do the ifft, we must reconstruct the other half of the fft signal
	fullSignal = np.concatenate((signal, reversedArray), axis=1)
	
	return fullSignal
	