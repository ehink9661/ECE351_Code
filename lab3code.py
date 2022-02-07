# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 2                                                        #
# Febuary 1st, 2022                                            #
# This lab deals with us creating user defined functions and   #
# using them to create plots of our functions and usderstand   #
# how time scaling affects the plots and functions             #
#                                                              #
################################################################

"""
Created on Tue Feb  1 13:35:28 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

steps = 0.01
xmin = 0.0
xmax = 20.0
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

# ramp function
def ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

# f1 function
def f1(t):
    y = step(t-2)-step(t-9)
    return y

# f2 function
def f2(t):
    y = np.exp(-t)*step(t)
    return y

# f3 function
def f3(t):
    y = ramp(t-2)*(step(t-2)-step(t-3))+ramp(4-t)*(step(t-3)-step(t-4))
    return y

y1 = f1(t)
plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, y1)
plt.grid()
plt.title('fx(t) Plots')
plt.ylabel('f1(t)')

y2 = f2(t)
plt.subplot (3,1,2)
plt.plot(t, y2)
plt.grid()
plt.ylabel('f2(t)')

y3 = f3(t)
plt.subplot (3,1,3)
plt.plot(t, y3)
plt.grid()
plt.ylabel('f3(t)')
plt.xlabel('t in Seconds')
plt.show()

# Part 2

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
    
t = np.arange(xmin, 2*(xmax+steps) + steps , steps) # double t size for the conv and add 1 step to the end to get everything

y = func_conv(y1, y2)
plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, y)
plt.title('Convolution via User Function')
plt.grid()
plt.ylabel('Convolution of f1 and f2')

y = func_conv(y2, y3)
plt.subplot (3,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('Convolution of f2 and f3')

y = func_conv(y1, y3)
plt.subplot (3,1,3)
plt.plot(t, y)
plt.grid()
plt.ylabel('Convolution of f1 and f3')
plt.xlabel('t in Seconds')
plt.show()

y = scipy.signal.convolve(y1, y2)
plt.figure(figsize = (10, 7))
plt.subplot (3,1,1)
plt.plot(t, y)
plt.title('Convolution via scipy.signal.convolve()')
plt.grid()
plt.ylabel('Convolution of f1 and f2')

y = scipy.signal.convolve(y2, y3)
plt.subplot (3,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('Convolution of f2 and f3')

y = scipy.signal.convolve(y1, y3)
plt.subplot (3,1,3)
plt.plot(t, y)
plt.grid()
plt.ylabel('Convolution of f1 and f3')

plt.xlabel('t in Seconds')
plt.show()