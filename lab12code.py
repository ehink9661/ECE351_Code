# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 12                                                       #
# April 12th, 2022                                             #
# This lab is our final project for ECE 351. It is based       #
# around using the python plotting skills to aid in creating   #
# a filter for a theoretical aircraft positioning control      #
# system.                                                      #
#                                                              #
################################################################
"""
Created on Tue Apr 12 13:30:44 2022

@author: Anaconda
"""

# the other packages you import will go here
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack  
import pandas as pd
import control as con

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0']. values
sensor_sig = df['1']. values

plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

# your code starts here , good luck

# task 1 use lab 8 and 9 for fast fourier transfomrs

def usrfftclean(x, fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N    # compute the frequencies for the output
                                        # signal , (fs is the sampling frequency and
                                        # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    for i in range(0, len(X_mag)):
        if(X_mag[i] < np.exp(-10)):
            X_phi[i] = 0
    return freq, X_mag, X_phi
    # ----- End of user defined function ----- #
    
fs = 1e6

def make_stem(ax ,x,y,color='k',style='solid',label='',linewidths =2.5 ,** kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x, 0 ,y, color=color , linestyles=style , label=label , linewidths=linewidths)
    ax.set_ylim ([1.05*y.min(), 1.05*y.max()])

    

freq, X_mag, X_phi = usrfftclean(sensor_sig, fs)

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.xlim(0, 2e5)
plt.grid()
plt.title('Magnitude of Noisy Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V]')
plt.show()

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_phi)
plt.xlim(0, 2e5)
plt.grid()
plt.title('Phase of Noisy Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [deg]')
plt.show()

# task 2 use lab 10 for RLC bandpass schematic


R = 1e2
L = 3.37e-3
C = 2.085e-6

steps = 1e1
xmin = 1e1
xmax = 2e6
w = np.arange(xmin, xmax + steps, steps)

num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

w, mag, phase = scipy.signal.bode((num, den), w)

sys = con.TransferFunction(num, den)
_ = con.bode(sys , w , dB = True , Hz = True , deg = True , plot = True)
plt.show()

_ = con.bode(sys , w , dB = True , Hz = True , deg = True , plot = True)
plt.xlim(10, 1800)
plt.show()

_ = con.bode(sys , w , dB = True , Hz = True , deg = True , plot = True)
plt.xlim(1800, 2000)
plt.subplot(2, 1, 1)
plt.ylim(-.5, 0)
plt.show()

_ = con.bode(sys , w , dB = True , Hz = True , deg = True , plot = True)
plt.xlim(2000, 100000)
plt.show()

_ = con.bode(sys , w , dB = True , Hz = True , deg = True , plot = True)
plt.xlim(100000, 200000)
plt.show()

numz, denz = scipy.signal.bilinear(num, den, fs)

z = scipy.signal.lfilter(numz, denz, sensor_sig)

plt.figure(figsize = (10, 7))
plt.plot(t, z)
plt.grid()
plt.title('Filtered Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

freq, X_mag, X_phi = usrfftclean(z, fs)

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.xlim(0, 2e5)
plt.grid()
plt.title('Magnitude of Noisy Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V]')
plt.show()

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_phi)
plt.xlim(0, 2e5)
plt.grid()
plt.title('Phase of Noisy Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [deg]')
plt.show()

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.xlim(0, 1800)
plt.ylim(0, 0.4)
plt.grid()
plt.title('Magnitude of Noisy Signal (0Hz - 1800Hz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V]')
plt.show()

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.xlim(1800, 2000)
plt.grid()
plt.title('Magnitude of Noisy Signal (1800Hz - 2000Hz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V]')
plt.show()

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.xlim(2000, 100000)
plt.ylim(0, 0.4)
plt.grid()
plt.title('Magnitude of Noisy Signal (2000Hz - 100000Hz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V]')
plt.show()

fig , ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.xlim(100000, 200000)
plt.ylim(0, 0.1)
plt.grid()
plt.title('Magnitude of Noisy Signal (100000Hz - 200000Hz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V]')
plt.show()