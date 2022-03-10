# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 6                                                        #
# Febuary 22th, 2022                                           #
# This lab deals with using python libraries to solve our      #
# partial fraction expansions for us, lowering our work time   #
# and learning a useful tool for more complex problems         #
#                                                              #
################################################################
"""
Created on Tue Feb 22 13:28:33 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math

steps = 0.01
xmin = 0.00
xmax = 2.00
t = np.arange(xmin, xmax + steps , steps)

# step function
def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

# Part 1

# Task 1

def y1(t):
    y = (0.5 - 0.5*np.exp(-4*t)+ np.exp(-6*t))*step(t)
    return y

y = y1(t)
plt.figure(figsize = (10, 7))
plt.subplot (2,1,1)
plt.plot(t, y)
plt.grid()
plt.title('Step Response')
plt.ylabel('y(t) Hand Calculated')

# Task 2

den = [1, 10, 24]
num = [1, 6, 12]
tout, yout = scipy.signal.step((num, den), T = t)
plt.subplot (2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('y(t) scipy.signal.step')
plt.xlabel('t in Seconds')
plt.show()

# Task 3

rout, pout, kout = scipy.signal.residue([0, 1, 6, 12], [1, 10, 24, 0])
print('part 1 task 3')
print('r =', rout)
print('p =', pout)
print('k =', kout)

# Part 2

# Task 1

rout, pout, kout = scipy.signal.residue([0, 0, 0, 0, 0, 0, 25250], [1, 18, 218, 2036, 9085, 25250, 0])
print('part 2 task 1')
print('r =', rout)
print('p =', pout)
print('k =', kout)

# task 2

steps = 0.01
xmin = 0.00
xmax = 4.50
t = np.arange(xmin, xmax + steps , steps)

def y2(t):
    y = ((-0.21461963*np.exp(-10*t)) + (2*0.8754*np.exp(-3*t)*np.cos(4*t + math.radians(123.69))) + (2*0.1044*np.exp(-1*t)*np.cos(10*t + math.radians(-27.16))) + 1)*step(t)
    return y

y = y2(t)
plt.figure(figsize = (10, 7))
plt.subplot (2,1,1)
plt.plot(t, y)
plt.grid()
plt.title('Time Domain Response')
plt.ylabel('y(t) Based on scipy.signal.residue')

# task 3

den = [1, 18, 218, 2036, 9085, 25250]
num = [ 0, 0, 0, 0, 0, 25250]
tout, yout = scipy.signal.step((num, den), T = t)
plt.subplot (2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('y(t) scipy.signal.step')
plt.xlabel('t in Seconds')
plt.show()