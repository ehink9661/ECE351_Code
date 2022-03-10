# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:36:02 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#task 1

def ak(k):
    a = 0*k
    return a

def bk(k):
    if k == 0:
        return math.inf # when k is 0, get und
    b = 2/(k*np.pi)*(1-np.cos(k*np.pi))
    return b

print('a0 = ', ak(0))
print('a1 = ', ak(1))
print('b1 = ', bk(1))
print('b2 = ', bk(2))
print('b3 = ', bk(3))

#task 2

T = 8

def x(n, t):
    x = np.zeros(t.shape)
    for i in range(1, n+1):
        x += bk(i)*np.sin(i*(2*np.pi/T)*t)
    return x

steps = 0.001
xmin = 0.00
xmax = 20.00
t = np.arange(xmin, xmax + steps , steps)

y = x(1, t)
plt.figure(figsize = (10, 7))
plt.subplot (3,2,1)
plt.plot(t, y)
plt.grid()
plt.title('Fourier Series Approximation')
plt.ylabel('N = 1')

y = x(3, t)
plt.subplot (3,2,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('N = 3')

y = x(15, t)
plt.subplot (3,2,3)
plt.plot(t, y)
plt.grid()
plt.ylabel('N = 15')

y = x(50, t)
plt.subplot (3,2,4)
plt.plot(t, y)
plt.grid()
plt.ylabel('N = 50')

y = x(150, t)
plt.subplot (3,2,5)
plt.plot(t, y)
plt.grid()
plt.xlabel('t in seconds')
plt.ylabel('N = 150')

y = x(1500, t)
plt.subplot (3,2,6)
plt.plot(t, y)
plt.grid()
plt.xlabel('t in seconds')
plt.ylabel('N = 1500')