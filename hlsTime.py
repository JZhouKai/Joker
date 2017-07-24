''' 
function for applying hearing loss to a signal in time domain 
'''


from __future__ import division
import math
import regalia_mitra


# def hls(inputSignal, Fs, f1, bw1, k1, f2, bw2, k2, f3, bw3, k3):
def hls(inputSignal):
	# hearing loss parameters (for the hls function)
	Fs = 16000
	
	# define parameters
	f1 = 2000
	bw1 = 500
	k1 = -10
	f2 = 4000
	bw2 = 800
	k2 = -30
	f3 = 7000
	bw3 = 1000
	k3 = -10

	# scaling parameters
	w1 = 2*math.pi*f1/Fs;      
	BW1 = 2*math.pi*bw1/Fs;   
	K1 = 10**(k1/20); #db2mag
	w2 = 2*math.pi*f2/Fs;      
	BW2 = 2*math.pi*bw2/Fs;    
	K2 = 10**(k2/20);        
	w3 = 2*math.pi*f3/Fs;     
	BW3 = 2*math.pi*bw3/Fs;    
	K3 = 10**(k3/20); 

	firstSimulation = regalia_mitra.regalia_mitra(inputSignal, w1, K1, BW1);
	secondSimulation = regalia_mitra.regalia_mitra(firstSimulation, w2, K2, BW2);
	simulatedSignal = regalia_mitra.regalia_mitra(secondSimulation, w3, K3, BW3);

	return simulatedSignal
	
	
	
	