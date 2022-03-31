# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 10                                                       #
# March 29th, 2022                                             #
# This lab deals with us using python to become familiar with  #
# frequency response tools and using bodi plots, as well as    #
# utilizing filters in our analysis.                           #
#                                                              #
################################################################
"""
Created on Tue Mar 29 13:34:15 2022

@author: Anaconda
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import control as con

# part 1

R = 1e3
L = 27e-3
C = 100e-9

steps = 1e3
xmin = 1e3
xmax = 1e6
w = np.arange(xmin, xmax + steps, steps)

def hmag (w):
    y = (w/(R*C)) / np.sqrt(w**4 + ((1/(R*C))**2 - 2/(L*C))*w**2 + (1/(L*C))**2)
    y = 20*np.log10(y) # convert to dB
    return y

def hdeg (w):
    y = np.pi/2 - np.arctan((w/(R*C))/((1/(L*C)-w**2)))
    y = y*180/np.pi
    return y

# task 1

y1 = hmag(w)

plt.figure(figsize = (10, 7))
plt.subplot (2,1,1)
plt.semilogx(w, y1)
plt.title('Part 1 - Task 1')
plt.grid()
plt.ylabel('|H| dB')

y2 = hdeg(w)
for i in range(len(y2)):
    if y2[i] > 90:
        y2[i] = y2[i] - 180 #  centers around 0

plt.subplot (2,1,2)
plt.semilogx(w, y2)
plt.grid()
plt.xlabel('rad/s')
plt.ylabel('/_H dB')
plt.show()

# task 2

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

w, mag, phase = scipy.signal.bode((num, den), w)

plt.figure(figsize = (10, 7))
plt.subplot (2,1,1)
plt.semilogx(w, mag)    # Bode magnitude plot
plt.title('Part 1 - Task 2')
plt.grid()
plt.ylabel('|H| dB')

plt.subplot (2,1,2)
plt.semilogx(w, phase)  # Bode phase plot
plt.grid()
plt.xlabel('rad/s')
plt.ylabel('/_H dB')
plt.show()

# task 3

sys = con.TransferFunction(num, den)
_ = con.bode(sys , w , dB = True , Hz = True , deg = True , plot = True)
# use _ = ... to suppress the output

# part 2

# task 1

def x(t):
    return np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

fs = 2*np.pi*50000

steps = 1/fs
xmin = 0
xmax = 1e-2
t = np.arange(xmin, xmax + steps, steps)

y = x(t)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.title('Part 2 - Task 1')
plt.grid()
plt.ylabel('|H| dB')

# task 2

numz, denz = scipy.signal.bilinear(num, den, fs)

# task 3

z = scipy.signal.lfilter(numz, denz, y)

# task 4

plt.figure(figsize = (10, 7))
plt.plot(t, z)
plt.title('Part 2 - Task 4')
plt.grid()
plt.ylabel('|H| dB')