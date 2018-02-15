# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:01:16 2018

@author: twiefel
"""

import matplotlib.pyplot as plt
import numpy as np
import requests
import copy
from cPickle import dumps, loads
class UltraschallCient:
    def __init__(self):
        pass
    def acquire(self):
        filename =  "hannes28"
        with open(filename+".dat", 'rb') as device:
            raw_data = device.read()
        print type(raw_data)
        print len(raw_data)
        print raw_data[0:10]
        Bytes_old = np.fromfile(filename+".dat", dtype = '<i4')
        print type(Bytes_old)
        print len(Bytes_old)
        print Bytes_old[0:10]
        url = "http://localhost:8000"
        r = requests.get(url,stream=True)
        #print r.content
        data = loads(r.content)
        print type(data)
        print len(data)
        print data[0:10]
        Bytes = np.fromstring(data, dtype = '<i4')
        print type(Bytes)
        print len(Bytes)
        print Bytes[0:10]
        
        
    def fast_image(self):
        filename = "live_image"
        url = "http://raspberrypi.local:8000"
        r = requests.get(url,stream=True)
        #print r.content
        data = loads(r.content)
        print type(data)
        print len(data)
        print data[0:10]
        Bytes = np.fromstring(data, dtype = '<i4')
        
        #real code starts here
        #Bytes = np.fromfile(filename+".dat", dtype = '<i4')
        bytes = Bytes[:-1]
    
        #ADC2_GPIO = [7, 8,9,10,11,23,24,25]
        #we want to have the bits from 7 to 8 and 23 to 25, so lets do som bit shifting magic...
        sbytes = copy.deepcopy(bytes)
        sbytes = np.right_shift(bytes,7) #so remove right bits below 7
        sbytes = np.bitwise_and(sbytes,31)
    
        bbytes = copy.deepcopy(bytes)
        bbytes = np.right_shift(bbytes,18) #so remove right bits below 7
        bbytes = np.bitwise_and(bbytes,511)
    
        mbytes = sbytes+bbytes
    
    
        #et voila
        M = mbytes
    
        #get sample frequency, duration and the time
        n = len(bytes)
        Duration = Bytes[-1]
        Fech = n*1.0/(Duration*1e-9)/1e6 # 1e-9 because Duration is in ns, 1e6 because Fech in Msps
        print "Duration:",Duration
        print "Fech:",Fech
        t = range(n)
        for k in range(n):
            t[k] = 1.0*t[k]/Fech
        print t[:10]
    
        rawSig = M
        #rawSig = M - np.average(M)
        print len(rawSig)
        
        T = t
        print T[:10]
        
        repeat_size = 400 #repeat size
        sample_size = 5000 #sample size
    
        FH = rawSig
    
        #reshape envelope of the signal
        tableData = np.asarray(FH).reshape((repeat_size,sample_size))
        
        IndexEmpty = 20 #where does this number come from?
        IndexLine = 104 #where does this number come from?
        
        ExLine = tableData[IndexLine]
        ExLineRaw = tableData[IndexLine]
        plt.figure(figsize=(15,5))
        plt.subplot(211)
        plt.plot(T[0:3000],rawSig[5000*IndexLine:5000*IndexLine+3000],"y", label='Raw signal')
        #plt.plot(T[0:3000],F[5000*IndexLine:5000*IndexLine+3000],"r", label='Filtered signal')
        plt.plot(T[0:3000],ExLine[0:3000],"b", label='Enveloppe of the signal')
        print len(T[0:3000])
        print len(ExLine[0:3000])
        #the dimensions are not matching. guess theres something wrong with the dimensions of tableData
        plt.title("Details of a line from "+filename.split("/")[-1])
        plt.xlabel("Time in uS")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.savefig('Imgs/ProcessingLine_'+filename.split("/")[-1]+".png", bbox_inches='tight')
        #plt.show()
        
        #add an offset to the image, will do that later
        #get average value
        Val = np.average(tableData)
        #tableData = np.asarray(FH).reshape((1000,2*2500))
        Offset = 400
        MinTable = 10*np.min(tableData)
        Zeroes = np.zeros((repeat_size,Offset))+Val
        BigTable = []
        BigTable = np.append(Zeroes, tableData, axis=1)
        
        plt.subplot(212)
        #plot the reshaped data
        #there are some dots, so guess the file contains data
        plt.imshow((abs(BigTable)), aspect='auto')
        plt.title("Image of "+filename.split("/")[-1])
        plt.axhline(IndexLine, color='r', linestyle='--')
        #plt.title("Mapping the data from "+RawData.split("/")[-1]+" .")  
        #plt.savefig('Imgs/map_'+filename.split("/")[-1]+".png", bbox_inches='tight')
        #
        plt.show()
        
    
if __name__ == "__main__":
    print "server started"
    uc = UltraschallCient()
    #uc.fast_image("hannes28")
    
    while True:
        uc.fast_image()