"""
takes an multi-channel audio file and outputs a beamformed audio file
http://www.labbookpages.co.uk/audio/beamforming/delayCalc.html

@author: kylesaltmarsh
"""

import math
import numpy as np

class Beamform:
    def __init__(self, filename, output_filename, sample_rate, incident_angle, spacing):
        self.filename = filename
        self.output_filename = output_filename

        self.delta_p = math.floor(sample_rate*spacing*math.sin(math.radians(incident_angle))/343.0)

    def beamforming(self):
        # read in the audio file
        audio = np.genfromtxt(self.filename, delimiter=',')

        # get the number of channels
        num_channels = audio.shape[1]

        beamformed_audio = np.zeros(audio.shape[0])
        for i in range(0,num_channels):
            channel_temp = np.append(np.zeros(self.delta_p*i), audio[0:(audio.shape[0]-self.delta_p*i),i])
            beamformed_audio = beamformed_audio + channel_temp

        # write the beamformed audio to a file
        np.savetxt(self.output_filename, beamformed_audio, delimiter=',')

if __name__ == "__main__":
    filename = 'test.csv'
    output_filename = 'beamformed.csv'
    sample_rate = 10000
    incident_angle = 45
    spacing = 0.1

    Beamform(filename, output_filename, sample_rate, incident_angle, spacing).beamforming()


            

