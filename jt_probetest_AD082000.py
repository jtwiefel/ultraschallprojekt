
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import griddata
import math
from scipy.signal import decimate, convolve
import copy

#name of file with raw data
#data_filename = "rawdata/felix04.dat"
data_filename = "rawdata/hannes12.dat"
#data_filename = "rawdata/probeX.data"
#data_filename = "rawdata/felix.data"

#pins of the AD082000
ADC2_GPIO = [7, 8,9,10,11,23,24,25]
ADC2len = len(ADC2_GPIO)

#ADC2_GPIO = [20,26,16,19,13,12, 7, 8,11]
#ADC2len = len(ADC2_GPIO)



def get_signal_details(time, data_size):
    T = [ (( x * time ) / ( 1000.0*data_size )) for x in 2*range(data_size)]
    Tt = [ (( x * time ) / ( 1000.0*data_size )) for x in range(data_size)]
    Fech = 1000.0*data_size/time # in MHz
    print "Fech",Fech
    print "len(Tt)",len(Tt)
    return T,Tt,Fech

"""
returns the voltages as a signal and a map
"""
def get_voltages(byte_voltages):
    Signal = []
    Map = np.zeros((len(byte_voltages),ADC2len), dtype=np.int)
    for i in range(len(byte_voltages)):
        val = byte_voltages[i]
        SignalZero = 0
        for k in range(ADC2len):
            Map[i][k] = (val & 2**k)/2**k
        for k in range(ADC2len):
            SignalZero += 2**k*((val & 2**ADC2_GPIO[k])/2**ADC2_GPIO[k])
        Signal.append(SignalZero)

            
    return Signal,Map



def plot_voltages_map(voltages_map):
    im = plt.imshow(voltages_map, cmap='hot', aspect="auto")
    plt.show()
    #plt.colorbar(voltages_map, orientation='horizontal')
    plt.plot(np.var(voltages_map,0))
    plt.show()
    for m in range(len(ADC2_GPIO)):
        if (np.var(voltages_map,0)[m] > 0.0003) :
            print str(m)+" - "+str(np.var(voltages_map,0)[m])


def normalize_voltages(voltages):
    avg_voltage = np.average(voltages)
    norm_voltages = voltages - avg_voltage
    return norm_voltages
    
def perform_fourier_transform(norm_voltages):
    print "performing FFT on raw signal"

    FFT = np.fft.fft(norm_voltages)
    
    #this stuff is to clean the signal, we do that later
    FFTCleaned = copy.deepcopy(FFT)
    FStart = 0.068*len(FFTCleaned)
    FStop = 0.196*len(FFTCleaned)
    

    
    for k in range(len(FFTCleaned)/2):
        if (k < FStart or k > FStop): # in (k < 550000 or k > 790000) # 0.068 0.196
            FFTCleaned[k] = 0
            FFTCleaned[-k] = 0
        
    print "FFT shape:",FFT.shape
    return FFT,FFTCleaned

def plot_fourier_transform(FFT_voltages,clean_FFT_voltages, frequency):
    Scale = max(FFT_voltages)
    num_values_per_step = 2
    ff = [ frequency*2.0*x/(len(FFT_voltages)) for x in range(len(FFT_voltages)/2)]


    print "In[44]:"
    
    freq = np.fft.fftfreq(len(FFT_voltages), d=1.0)
    print freq
    
    plt.figure(figsize=(15,5))
    #plt.plot(freq,FFT_voltages,"g")
    #plt.plot(np.real(FFT_voltages)[0:len(FFT_voltages)/2]/Scale,"b")
    #
    plt.plot(ff,np.real(FFT_voltages)[0:len(FFT_voltages)/2]/Scale,"b")
    plt.plot(ff,np.imag(FFT_voltages)[0:len(FFT_voltages)/2]/Scale,"g")
    plt.plot(ff,np.imag(clean_FFT_voltages)[0:len(FFT_voltages)/2]/Scale,"y")
    plt.plot(ff,np.real(clean_FFT_voltages)[0:len(FFT_voltages)/2]/Scale,"y")
    plt.title("Details of the FFT of the data from "+data_filename.split("/")[-1]+" .")  
    plt.xlabel("Frequency (MHz)")
    plt.savefig('Imgs/fft_'+data_filename.split("/")[-1]+".png", bbox_inches='tight')
    plt.show()

def CreateSC(RawImgData,Val):
    LenLinesC = np.shape(RawImgData)[1]
    NbLinesC = np.shape(RawImgData)[0]
    SC = np.zeros((LenLinesC,LenLinesC))+Val
    SC += 1
    maxAngle = 60.0
    step = maxAngle/(NbLinesC+1)
    CosAngle = math.cos(math.radians(30))
    Limit = LenLinesC*CosAngle

    points = []
    values = []

    for i in range(LenLinesC):
        for j in range(LenLinesC):
            if (  (j > LenLinesC/2 + i/(2*CosAngle)) or  (j < LenLinesC/2 - i/(2*CosAngle)) ):
                SC[i][j] = 0
                points.append([i,j])
                values.append(0)
            if (  (i > Limit) ):
                if ( (i**2 + (j-LenLinesC/2) ** 2) > LenLinesC**2):
                    SC[i][j] = 0 
                    points.append([i,j])
                    values.append(0)
    for i in range(NbLinesC):
        PointAngle = i*step-30
        COS = math.cos(math.radians(PointAngle))
        SIN = math.sin(math.radians(PointAngle))
        for j in range(LenLinesC):

            X = (int)( j*COS)
            Y = (int)(LenLinesC/2 - j*SIN)
            SC[X][Y] = RawImgData[i][j]
            points.append([X,Y])
            values.append(RawImgData[i][j])

    values = np.array(values,dtype=np.int)
    
    return SC,values,points,LenLinesC

"""
main stuff to do
"""
def main():
    
##t_new,V_new,T_new,Tt_new,Fech_new = get_data_from_file(RawData_new,dtype='<i4')
#t,V,T,Tt,Fech = get_data_from_file(RawData_new,dtype='<i4')
#print V[:]
#
    byte_data = np.fromfile(data_filename, dtype = '<i4')
    #f = open(data_filename, "r")
    #byte_data = np.fromfile(f, dtype=np.uint32)    
    print "data points:", len(byte_data)
    
    byte_voltages = byte_data[:-1]
    time = byte_data[-1]
    #print time
    T,Tt,frequency =  get_signal_details(time,len(byte_voltages))
    
    
    voltages_signal, voltages_map = get_voltages(byte_voltages)
    



    
    print voltages_map
    print voltages_map.shape
    
    #plot_voltages_map(voltages_map)
    #plt.plot(voltages_map)
    norm_voltages = normalize_voltages(voltages_signal)
    
    rawSig = []
    for k in range(len(norm_voltages)):
        #rawSig.append(Mb[k])
        rawSig.append(norm_voltages[k])
        #rawSig.append(norm_voltages[k])
 
    
    #plt.plot(voltages_signal)
    #plt.show()
    
    #get the freq spectrum and filter out unwanted freqs
    FFT_voltages,clean_FFT_voltages = perform_fourier_transform(rawSig)
    print FFT_voltages.shape
    #plot_fourier_transform(FFT_voltages,clean_FFT_voltages,frequency)
    #plt.plot(FFT_voltages)
    #plt.show()
    
    #transform the fft back to a time signal
    F = np.real(np.fft.ifft(FFT_voltages))
    #F = rawSig
    
    #calculate the envelope using hilbert transform
    FH = np.asarray(np.abs(signal.hilbert(F)))

    #num_adc = 1 
    
    #tableData = np.asarray(FH).reshape((1000,num_adc*10))

    #reshape the table in new dimensions.
    #i don't know where the number 1000 comes from...
    #xdim = 2000
    #ydim = (len(byte_data)-1)/xdim
    ydim = 2500
    xdim = (len(byte_data)-1)/ydim
    tableData = np.asarray(FH).reshape((xdim,ydim))

    #get the average signal value
    Val = np.average(tableData)
    
    #add an offset with the avg signal value.
    #i guess this has only optical reasons
#    #Offset = 400
#    Offset = 5
#    #MinTable = 10*np.min(tableData)
#    Zeroes = np.zeros((1000,Offset))+Val
#    BigTable = []
#    BigTable = np.append(Zeroes, tableData, axis=1)
#    tableData = BigTable[:,:3000+Offset]
    
    # In[51]:
    
    
#
#    plt.imshow((abs(tableData)), aspect='auto')
#    #plt.axhline(IndexLine, color='r', linestyle='--')
#    plt.title("Mapping the data from "+data_filename.split("/")[-1]+" .")  
#    plt.savefig('Imgs/map_'+data_filename.split("/")[-1]+".png", bbox_inches='tight')
#    plt.show()
    
    #tableData = BigTable[:,:3000+Offset]
    plt.imshow((abs(tableData)), aspect='auto')
    #plt.axhline(IndexLine, color='r', linestyle='--')
    #plt.title("Mapping the data from "+RawData.split("/")[-1]+" .")  
    #plt.savefig('Imgs/map_'+RawData.split("/")[-1]+".png", bbox_inches='tight')
    plt.show()
    return
    #these are the index lines.
    #the values have to be found manually
    ListOfPoints= [2920, 8682]
    #ListOfPoints= [104, 418, 741]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    
    TmpImg = (abs(tableData[ListOfPoints[0]-100:ListOfPoints[0]+150]))**(1.1)
    ax1.imshow(TmpImg,cmap=plt.get_cmap('gray'), aspect='auto')
    ax1.axhline(100, color='r', linestyle='--')
    TmpImg = (abs(tableData[ListOfPoints[1]-100:ListOfPoints[1]+150]))**(1.1)
    ax2.imshow(TmpImg,cmap=plt.get_cmap('gray'), aspect='auto')
    ax2.axhline(100, color='r', linestyle='--')
#    TmpImg = (abs(tableData[ListOfPoints[2]-100:ListOfPoints[2]+150]))**(1.1)
#    ax3.imshow(TmpImg,cmap=plt.get_cmap('gray'), aspect='auto')
#    ax3.axhline(100, color='r', linestyle='--')
    plt.suptitle('The 3 images in the file')
    plt.savefig('Imgs/images_'+data_filename.split("/")[-1]+".png", bbox_inches='tight')
    #plt.savefig('Imgs/mapCleanImage_'+RawData.split("/")[-1]+str(Start)+"-"+str(Stop)+".jpg", bbox_inches='tight')
    plt.show()
    
    
    
    #adds up all images from the 3 index lines
    #guess this is used to have some kind of average??
    DecImg = []
    for i in range(150):
        tmp = decimate(tableData[ListOfPoints[0]-70+i], 5, ftype='fir')
        #print tmp.shape
        #tmp += decimate(tableData[ListOfPoints[1]-70+i], 5, ftype='fir')
        #tmp += decimate(tableData[ListOfPoints[2]-70+i], 5, ftype='fir')
        #print tmp.shape
        #print tmp.shape
        #raw_input()
        DecImg.append(tmp)
        #SmallImg = DecImg
        
    #downsamples by factor 2
    SmallImg = []
    for i in range(len(DecImg)/2):
        SmallImg.append((DecImg[2*i]+DecImg[2*i+1])/2)
        
    #create the rotated shrinked image.
    #I guess this is not necessary in the end
    SCH,valuesH,pointsH,LenLinesCH = CreateSC(SmallImg,Val)
    grid_xH, grid_yH = np.mgrid[0:LenLinesCH:1, 0:LenLinesCH:1]
    grid_z1H = griddata(pointsH, valuesH, (grid_xH, grid_yH), method='linear')
    plt.figure(figsize=(10,10))
    plt.imshow((grid_z1H**0.7),cmap=plt.get_cmap('gray')) 
    plt.title("Getting the image out of the data file: "+data_filename.split("/")[-1]+" .")  
    plt.savefig('Imgs/pic_'+data_filename.split("/")[-1]+".png", bbox_inches='tight')
    plt.show()
    #plt.plot(norm_voltages)
    #plt.plot(voltages_signal)
    #plt.show()
    #plt.show()
    #plot_voltages_map(voltages_map)
    
    

if __name__ == "__main__":
    print "Welcome"
    main()
    

