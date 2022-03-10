# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 4                                                        #
# Febuary 8th, 2022                                            #
# This lab deals with using convolutions to calculate a        #
# systems step response. All the code worked as expected       #
# except for h1(t) which seems to always drop at t=10 instead  #
# of where the step function dictates                          #
#                                                              #
################################################################

"""
Created on Tue Feb  1 13:35:28 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt
import math

steps = 0.1
xmin = -10.0
xmax = 10.0
t = np.arange(xmin, xmax + steps , steps)

# Part 1

# step function
def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

# convolution function
def func_conv(f1, f2):
    nf1 = len(f1) # get length of incoming arrays
    nf2 = len(f2)
    f1ext = np.append(f1, np.zeros((1, nf2-1))) # increases size to f1 + f2
    f2ext = np.append(f2, np.zeros((1, nf1-1)))
    result = np.zeros(f1ext.shape) # get array size of dataset filled with 0
    for i in range(nf2+nf1-2): 
        result[i] = 0
        for j in range(nf1):
            if (i-j+1)>0:
                try: 
                    result[i] += f1ext[j]*f2ext[i-j+1] # preform convolution for all values in f2 on a point in f1
                except:
                    print(i, j) # print if error occured
    return result

f = 0.25
w = 2*math.pi*f

# h1 function
def h1(t):
    y = np.exp(-2*t)*(step(t)-step(t-3))
    return y

# h2 function
def h2(t):
    y = step(t-2)-step(t-6)
    return y

# h3 function
def h3(t):
    y = (np.cos(w*t))*step(t)
    return y

y1 = h1(t)
plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, y1)
plt.grid()
plt.title('hx(t) Plots')
plt.ylabel('h1(t)')

y2 = h2(t)
plt.subplot (3,1,2)
plt.plot(t, y2)
plt.grid()
plt.ylabel('h2(t)')

y3 = h3(t)
plt.subplot (3,1,3)
plt.plot(t, y3)
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t in Seconds')
plt.show()

ystep = step(t)

t = np.arange(2*(xmin+steps), 2*(xmax+steps) + steps , steps)

y = func_conv(y1, ystep)*steps
plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, y)
plt.title('Step Response via User Functions')
plt.grid()
plt.ylabel('h1(t)')

y = func_conv(y2, ystep)*steps
plt.subplot (3,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('h2(t)')

y = func_conv(y3, ystep)*steps
plt.subplot (3,1,3)
plt.plot(t, y)
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t in Seconds')
plt.show()

y = 0.5*((1-np.exp(-2*t))*step(t)-(1-np.exp(-2*(t-3)))*step(t-3))
plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, y)
plt.title('Step Response via Hand Calculations')
plt.grid()
plt.ylabel('h1(t)')

y = (t-2)*step(t-2)-(t-6)*step(t-6)
plt.subplot (3,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('h2(t)')

y = 1/w*np.sin(w*t)*step(t)
plt.subplot (3,1,3)
plt.plot(t, y)
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t in Seconds')
plt.show()

