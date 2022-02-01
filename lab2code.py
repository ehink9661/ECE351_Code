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
Created on Tue Jan 25 13:35:59 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt

steps = 0.01
xmin = -5.0
xmax = 10.0
t = np.arange(xmin, xmax + steps , steps)

def func1(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y[i] = np.cos(t[i])
    return y
y = func1(t)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('cos(t) Function')
plt.ylabel('cos(t) Function')
plt.xlabel('t in Seconds')
plt.show()

# Part 2

# Task 2
t = np.arange(xmin, xmax + steps , steps)

def ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

y = ramp(t)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('ramp(t) Function')
plt.ylabel('ramp(t)')
plt.xlabel('t in Seconds')
plt.show()

def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

y = step(t)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('step(t) Function')
plt.ylabel('step(t)')
plt.xlabel('t in Seconds')
plt.show()

# Task 3
t = np.arange(xmin, xmax + steps , steps)

def f(t):
    y = 1*ramp(t) - 1*ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)
    return y

y = f(t)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Figure 2')
plt.ylabel('y(t)')
plt.xlabel('t in Seconds')
plt.show()

# Part 3

# Task 1
t = np.arange(xmin - 5.0, xmax - 5.0 + steps, steps)
y = f(-t)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Timescale of y(-t)')
plt.ylabel('f(-t)')
plt.xlabel('t in Seconds')
plt.show()

# Task 2
t = np.arange(xmin + 4.0, xmax + 4.0 + steps, steps)
y = f(t-4)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Timescale of y(t-4)')
plt.ylabel('y(t-4)')
plt.xlabel('t in Seconds')
plt.show()

t = np.arange(xmin - 9.0, xmax - 9.0 + steps, steps)
y = f(-t-4)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Timescale of y(-t-4)')
plt.ylabel('y(-t-4)')
plt.xlabel('t in Seconds')
plt.show()

# Task 3
t = np.arange(2*xmin, 2*xmax + steps, steps)
y = f(t/2)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Timescale of y(0.5t)')
plt.ylabel('y(t/2)')
plt.xlabel('t in Seconds')
plt.show()

t = np.arange(xmin/2, xmax/2 + steps, steps)
y = f(2*t)
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Timescale of y(2t)')
plt.ylabel('y(2*t)')
plt.xlabel('t in Seconds')
plt.show()

# Task 5
steps = 1.0 # derivative has 1 step size for easier visualization
t = np.arange(xmin, xmax + steps, steps)
y = np.diff(f(t-1))
t = np.arange(xmin, xmax, steps) # shrinks t for derivative graph
plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title('Differentiation of y(t)')
plt.ylabel('diff(y(t-1))')
plt.xlabel('t in Seconds')
plt.show()