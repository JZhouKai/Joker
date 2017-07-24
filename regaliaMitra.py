# regalia mitra code
# allpass filtering based on the Regalia Mitra model

# w0 : central frequency[rads/sample]
# k  : gain at f0 for boost/cut. (1 = original signal, 0<k<1 cut, >1 boost)
# bw : bandpass at -3dB [rads/sample]
import math
from scipy.signal import lfilter


def regalia_mitra(inputSignal, w0, k, bw):

	k1 = (1-math.tan(bw/2))/(1+math.tan(bw/2))
	k2 = -math.cos(w0)
	k0 = k/2

	b = [k1, k2*(1+k1), 1]
	a = [1, k2*(1+k1), k1]
 
	y = lfilter(b, a, inputSignal)
	outputSignal = 0.5*inputSignal + 0.5*y + k0*inputSignal - k0*y

	return outputSignal


