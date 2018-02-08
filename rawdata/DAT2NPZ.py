import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import decimate, convolve
import math
from scipy.interpolate import griddata

ADC2_GPIO = [7, 8,9,10,11,23,24,25]
ADC2len = len( ADC2_GPIO)

def GetSeries(Volts):
    
    Map = np.zeros((len(Volts)-1,34), dtype=np.int)#why 34 here?
    for i in range(len(Volts)-1):
        val = Volts[i]
        for k in range(34):
            Map[i][k] = (val & 2**k)/2**k
        if not (i%10000):
            print i
    return Map

def GetV(Volts):
    Signal = [] 
    for i in range(len(Volts)):
        if not (i%100000):
            print 100.0*i/len(Volts)
        val = Volts[i]
        SignalZero = 0 
        for k in range(ADC2len):
            SignalZero += 2**k*((val & 2**ADC2_GPIO[k])/2**ADC2_GPIO[k])
        Signal.append(SignalZero)

    return Signal

def GetV2(Volts):
    Signal = []
    Map = np.zeros((len(Volts),ADC2len), dtype=np.int)
    for i in range(len(Volts)):
        if not (i%100000):
            print 100.0*i/len(Volts)
        val = Volts[i]
        SignalZero = 0
        for k in range(ADC2len):
            Map[i][k] = (val & 2**k)/2**k
        for k in range(ADC2len):
            SignalZero += 2**k*((val & 2**ADC2_GPIO[k])/2**ADC2_GPIO[k])
        Signal.append(SignalZero)

    return Signal,Map


filename = "hannes12.dat"

def CreateNPZ(filename):
	Bytes = np.fromfile(filename, dtype = '<i4')

	M = GetV(Bytes)


	#n = len(Bytes)
	n = len(Bytes)-1#did you mean this?
	print n
	#Map = GetSeries(Bytes)
	Duration = Bytes[-1]
	print Duration
	#Map = GetSeries(Bytes)#did you mean this?
	Fech = n*1.0/(Duration*1e-9)/1e6 # 1e-9 because Duration is in ns, 1e6 because Fech in Msps
	print Fech

	t = range(n)
	for k in range(n):
	    t[k] = 1.0*t[k]/Fech
	#M = GetV2(Bytes)[0]
	print "go"
	#M = GetV2(Bytes[:-1])[0]#did you mean this?

	plt.plot(t,M[0:n])
	plt.xlabel('t in us')
	plt.title("File: "+filename+". Sampling freq is "+str(Fech)+" Msps.")
	plt.savefig(filename.split(".")[0]+'-all.png')  
	plt.show()
	plt.clf()
	plt.cla()
	plt.close()
	np.savez_compressed(filename.split(".")[0]+'.npz', M=M, t=t, Fech=Fech, Duration=Duration)

if __name__ == '__main__':
    import sys
    print sys.argv[1]
    CreateNPZ(sys.argv[1])
