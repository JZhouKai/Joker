'''
define own objectives in order to add hearing losses to the signal

'''


import numpy as np
np.random.seed(1337)  # for reproducibility

# import hls
	
# function to round up numbers to the nearest odd number. 
# "quantizing"
def roundUp(array):
    return np.ceil(array) // 2 * 2 + 1

# function to expand input signal
def expand(array):
	
	return (2*array)
	# return [ 2*array if array > 30 else 0]

	
# defining the loss function which should include the hearing loss 
# the inputs to this function are "tensorType(float32, matrix)". 
def lossFunction(y_true, y_pred):

	# starting by just attenuating whole signal 
	edited = y_pred * 0.2
	
	# quantize
	#edited = roundUp(y_pred)
	
	# expand
	#edited = expand(y_pred)
	
	difference = abs(edited-y_true)
	return difference**2