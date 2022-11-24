"""
Created on Wed May 17 20:42:20 2017

@author: kylesaltmarsh
"""

import wavio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class InstrumentNoise:
    
    def get_audio(self, filename, start = 0, end = 0.1):
        """Obtain the certain time section audio data from the origin wav file.
        Parameters
        ----------
        filename: origin audio file
        start : float, start time of the wanted audio signal.
        end : float, end time of the wanted audio signal.
        """
        self.df = wavio.read(filename)
        self.start = start
        self.end = end
        self.rate = self.df.rate
        self.data = self.df.data[int(self.start * self.df.rate) : int(self.end * self.df.rate)]
        
    def cut_frame(self, data, window, step):
        """cut the data into multiple frames
        
        * Parameters:
            data: the data you want to cut into frame, the total length may change depends on
                  the window, step and data length
            window: the quantity of samples contain in a frame (ms)
            step: the sample time different between two adjacent frame (ms)
        * Example:
            newframe = self.cut_frame(df, 10, 5) #window length is 10 ms, step is 5 ms
        * Return:
            return the data frame in matrix style, each row is one frame
        """
        
        frame = []
        for i in range(int(len(data)/(step*self.rate/1000.0) - window/step+1)):
            frame.append(data[int(i*step*self.rate/1000):int((i*step+window)*self.rate/1000)])
        frame = np.array(frame)
        return frame
    
    def export_csv(self,filename):
        """export the post-processing data as csv file, set the minimum data to 0
        * Parameters:
            filename: the export time
        * Example:
            a.ExportWav('export')
        * Return:
            return nothing, just output the csv data file
        """
        temp = pd.DataFrame(self.data)
        temp.columns = ['Sample_Rate_' + str(self.rate)]
        temp.to_csv(filename+'.csv',index = None)
            
    def butter_pass(self, cutoff, btype, order):
        """create the zero and polar of the filter, use the butter filter in this function
          more filters will be added later
        
        * Parameters:
            cutoff: the frequency limit, for bandpass filter, the parameter length must be 2
            btype: choose the type of the filter
                   high: for high pass
                   low: for low pass filter
                   band: for band pass filter
            order: the order of the filter, higher the order is, sharper the filter is
        * Example:
            b, a = self.butter_pass([20,800], 'band', order=10) #create the coefficient of bandpass filter between 20 Hz to 800 Hz
        * Return:
            return the coefficient for filter
        """
        
        nyq = 0.5 * self.rate
        normal_cutoff = np.array(cutoff) / nyq
        if len(normal_cutoff) == 2:
            normal_cutoff = list(normal_cutoff)
        return butter(order, normal_cutoff, btype=btype, analog=False)

    def butter_pass_filter(self, cutoff, btype, order=10, inplace = False):
        """apply the filter to the self.data
        
        * Parameters:
            cutoff: the frequency limit, for bandpass filter, the parameter length must be 2
            btype: choose the type of the filter
                   "high": for high pass
                   "low": for low pass filter
                   "bandpass": for band pass filter
                   "bandstop": for band stop filter
            order: the order of the filter, higher the order is, sharper the filter is. the default value is 10
            inplace: use the filtered data as the self.data
        * Example:
            a.butter_pass_filter([20,800], "band", order=10, inplace = True)
        * Return:
            return the filtered data
        """

        b, a = self.butter_pass(cutoff, btype, order=order)
        filtereddata = lfilter(b, a, self.data.ravel())
        if inplace:
            self.data = filtereddata
        return filtereddata            

    def cal_ber(self, window = 30, step = 10, frel = [3000], freu = [3000], Enhance = True, batch = 20, AddFeature = False):
        """Band Energy Ratio
        """
        data = self.data.ravel()
        column = int(len(data)/(step*self.rate/1000.0) - window/step+1)
        rownum = len(frel)
        self.BER = np.zeros([column,rownum])
        freup = self.rate/2.0
        for i in range(0,column,batch):
            temp = []
            for k in range(i,i+min([batch,column - i])):
                temp.append(data[int(k*step*self.rate/1000):int((k*step+window)*self.rate/1000)])
            temp = np.array(temp)
            if np.abs(temp).sum():
                Spectrum = (abs(np.fft.rfft(temp) / float(temp.shape[1])) ** 2)
                SpeShape = Spectrum.shape[1]
                SpeSum = np.sqrt(np.sum(Spectrum,axis = 1))
                for j in range(rownum):
                    temp = Spectrum[:,int(frel[j] * SpeShape / freup):int(freu[j] * SpeShape / freup)+1]
                    self.BER[i:k+1,j] = np.sqrt(np.sum(temp, axis = 1)) / SpeSum
            else:
                self.BER[i:k+1,:] = 0
        if AddFeature:
            try:
                self.Feature = np.concatenate((self.Feature,self.BER.reshape(-1,1)), axis=1)
            except:
                self.Feature = self.BER.reshape(-1,1)
        return self.BER
        
    def cal_flatness(self, window = 30, step = 10, Enhance = True, batch = 20, AddFeature = False):
        """Flatness
        """
        data = self.data.ravel()
        column = int(len(data)/(step*self.rate / 1000.0) - window/step + 1)
        self.Flatness = np.zeros(column)
        if Enhance:
            AdjustNoiseSpectrum = np.interp(np.linspace(0,self.rate,window*self.rate/1000/2+1),np.linspace(0,self.rate,len(self.NoiseSpectrum)),self.NoiseSpectrum)
        else:
            AdjustNoiseSpectrum = np.ones(int(2000 * window / 1000))
        for i in range(0,column,batch):
            temp = []
            for k in range(i,i+min([batch,column - i])):
                temp.append(data[int(k*step*self.rate/1000):int((k*step+window)*self.rate/1000)])
            temp = np.array(temp)
            Spectrum = abs(np.fft.rfft(temp) / float(temp.shape[1]))[:,:int(2000 * window / 1000)] / AdjustNoiseSpectrum[:int(2000 * window / 1000)]
            self.Flatness[i:k+1] = np.exp(np.mean(np.log(Spectrum),axis = 1)) / np.mean(Spectrum,axis = 1)
        if AddFeature:
            try:
                self.Feature = np.concatenate((self.Feature,self.Flatness.reshape(-1,1)), axis=1)
            except:
                self.Feature = self.Flatness.reshape(-1,1)
        return self.Flatness
    
    def cal_hef(self, window = 50, step = 25, frel=100, AddFeature = False):
        """Highest Energy Frequency
        """
        data = self.data.ravel()
        frame = self.cut_frame(data, window, step)
        SpectrumFrame = abs(np.fft.rfft(frame) / float(frame.shape[1]))
        start = frel * SpectrumFrame.shape[1] / (self.rate/2.0)
        self.HEF = np.argmax(SpectrumFrame[:,int(start):],axis = 1) + start
        self.HEF = self.HEF* (self.rate/2.0) / SpectrumFrame.shape[1]
        if AddFeature:
            try:
                self.Feature = np.concatenate((self.Feature,self.HEF.reshape(-1,1)), axis=1)
            except:
                self.Feature = self.HEF.reshape(-1,1)
        return self.HEF 
    
    def cal_fcc(self, window = 50, step = 25, frel = 100, freu = 3000, AddFeature = False):
        """Frequency Centroid Centre
        """
        data = self.data.ravel()
        frame = self.cut_frame(data, window, step)
        SpectrumFrame = abs(np.fft.rfft(frame) / float(frame.shape[1])) ** 2
        fre = np.array([np.linspace(0, self.rate/2.0, SpectrumFrame.shape[1]),]*SpectrumFrame.shape[0]) 
        self.FCC = np.mean(SpectrumFrame[:,int(frel * SpectrumFrame.shape[1] / (self.rate/2.0)):int(freu * SpectrumFrame.shape[1] / (self.rate/2.0))] * fre[:,int(frel * SpectrumFrame.shape[1] / (self.rate/2.0)):int(freu * SpectrumFrame.shape[1] / (self.rate/2.0))], axis = 1) / np.mean(SpectrumFrame,axis = 1)
        if AddFeature:
            try:
                self.Feature = np.concatenate((self.Feature,self.FCC.reshape(-1,1)), axis=1)
            except:
                self.Feature = self.FCC.reshape(-1,1)
        return self.FCC
    
    def hoc(self, order = 2, visualize = False, window = 50, step = 25,batch = 10, AddFeature = False):
        """calculate the hoc in dataframe
        
        * Parameters:
            order: the order of the hoc 
            window: the quantity of samples contain in a frame (ms)
            visualize: visulize the result, default to be False
        * Example:
            Gate = self.hoc(order = 2, window = 10)
        * Return:
            return the hoc in array
        """
        data = self.data.ravel()
        for i in range(order):
            data = data[1:] - data[:-1]
        newdata = ((data[1:] * data[:-1]) < 0).astype(int)
        size = int(len(newdata)/(step*self.rate/1000.0)) - window/step+1
        self.hoc = np.zeros(size)
        for i in range(0,size,batch):
            temp = []
            for k in range(i,i+min([batch,size-i])):
                temp.append(newdata[k*step*self.rate/1000:(k*step+window)*self.rate/1000])
            self.hoc[i:k+1] = np.mean(temp,axis = 1)
            
        if AddFeature:
            try:
                self.Feature = np.concatenate((self.Feature,self.hoc.reshape(-1,1)), axis=1)
            except:
                self.Feature = self.hoc.reshape(-1,1)
        return self.hoc