# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 5                                                        #
# Febuary 15th, 2022                                           #
# This lab deals with solving for the impulse equation using   #
# hand calculated impulse equations using laplace, and using   #
# the python functions to solve for the impulse and the step   #
# response                                                     #
#                                                              #
################################################################
"""
Created on Tue Feb 15 13:31:42 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math

steps = 0.00001
xmin = 0.000
xmax = 0.0012
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

def h1(t):
    y = 10356*np.exp(-5000*t)*(np.sin(18585*t+math.radians(105)))*step(t)
    return y

y1 = h1(t)
plt.figure(figsize = (10, 7))
plt.subplot (2,1,1)
plt.plot(t, y1)
plt.grid()
plt.title('Inpulse Response')
plt.ylabel('h(t) Hand Calculated')

num = [0, 10000, 0]
den = [1, 10000, 370370370]
tout, yout = scipy.signal.impulse((num, den), T = t)
plt.subplot (2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('Using signal.impulse')
plt.xlabel('t in Seconds')
plt.show()

# Task 2

num = [0, 10000, 0]
den = [1, 10000, 370370370]
tout, yout = scipy.signal.step((num, den), T = t)
plt.plot(tout, yout)
plt.grid()
plt.title('Step Response')
plt.ylabel('Step response of H(s)')
plt.xlabel('t in Seconds')
plt.show()

