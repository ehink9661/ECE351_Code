# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:19:47 2022

@author: Anaconda
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import math

def usrfft(x, fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N    # compute the frequencies for the output
                                        # signal , (fs is the sampling frequency and
                                        # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    return freq, X_mag, X_phi
    # ----- End of user defined function ----- #

steps = 0.01
xmin = 0.00
xmax = 2.00
t = np.arange(xmin, xmax, steps)

def y1(t):
    y = np.cos(2*np.pi*t)
    return y

def y2(t):
    y = 5*np.sin(2*np.pi*t)
    return y

def y3(t):
    y = 2*np.cos((2*np.pi*2*t)-2) + np.sin((2*np.pi*6*t)+3)
    return y

x = y1(t)
fs = 100
freq, X_mag, X_phi = usrfft(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 1 - User Defined FFT of x(t)')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[96:105], X_mag[96:105]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[96:105], X_phi[96:105]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

x = y2(t)
fs = 100
freq, X_mag, X_phi = usrfft(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 2 - User Defined FFT of x(t)')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[96:105], X_mag[96:105]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[96:105], X_phi[96:105]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

x = y3(t)
fs = 100
freq, X_mag, X_phi = usrfft(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 3 - User Defined FFT of x(t)')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[86:115], X_mag[86:115]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[86:115], X_phi[86:115]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

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
    
x = y1(t)
fs = 100
freq, X_mag, X_phi = usrfftclean(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 4.1 - Clean User Defined FFT of x(t)')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[96:105], X_mag[96:105]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[96:105], X_phi[96:105]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

x = y2(t)
fs = 100
freq, X_mag, X_phi = usrfftclean(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 4.2 - Clean User Defined FFT of x(t)')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[96:105], X_mag[96:105]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[96:105], X_phi[96:105]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

x = y3(t)
fs = 100
freq, X_mag, X_phi = usrfftclean(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 4.3 - Clean User Defined FFT of x(t)')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[86:115], X_mag[86:115]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[86:115], X_phi[86:115]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

T = 8

def bk(k):
    if k == 0:
        return math.inf # when k is 0, get und
    b = 2/(k*np.pi)*(1-np.cos(k*np.pi))
    return b

def x(n, t):
    x = np.zeros(t.shape)
    for i in range(1, n+1):
        x += bk(i)*np.sin(i*(2*np.pi/T)*t)
    return x

steps = 0.01
xmin = 0.00
xmax = 16.00
t = np.arange(xmin, xmax, steps)

x = x(15, t)
fs = 100
freq, X_mag, X_phi = usrfftclean(x, fs)

plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, x)
plt.title('Task 5 - Clean User Defined FFT of x(t) Square Wave')
plt.grid()
plt.xlabel('t in sec')
plt.ylabel('x(t)')

plt.subplot (3,2,3)
plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot (3,2,4)
plt.stem(freq[760:841], X_mag[760:841]) # you will need to use stem to get these plots to be
plt.grid()

plt.subplot (3,2,5)
plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.ylabel('/_X(f)')

plt.subplot (3,2,6)
plt.stem(freq[760:841], X_phi[760:841]) # correct , remember to label all plots appropriately
plt.grid()
plt.xlabel('freq in Hz')
plt.show()

print(len(freq))