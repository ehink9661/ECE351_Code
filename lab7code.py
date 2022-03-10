# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 7                                                        #
# March 1st, 2022                                              #
# This lab deals with using python libraries to solve laplace  #
# transformation as well as partial fraction expansion to      #
# determine the stability of systems		               #
#                                                              #
################################################################
"""
Created on Tue Mar  1 13:43:38 2022

@author: Ethan Hinkle
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

steps = 0.01
xmin = 0.00
xmax = 10.00
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

# Task 2

zout, pout, kout = scipy.signal.tf2zpk([0, 0, 1, 9], [1, -2, -40, -64])
print('part 1 task 2 G(s)')
print('z =', zout)
print('p =', pout)
print('k =', kout)

zout, pout, kout = scipy.signal.tf2zpk([0, 1, 4], [1, 4, 3])
print('part 1 task 2 A(s)')
print('z =', zout)
print('p =', pout)
print('k =', kout)

zout = np.roots([1, 26, 168])
print('part 1 task 2 B(s)')
print('z =', zout)

# task 5

num = scipy.signal.convolve([1, 4], [1, 9])
den = scipy.signal.convolve([1, 4, 3], [1, -2, -40, -64])
print('part 1 task 5')
print('numerator', num)
print('denominator', den)
tout, yout = scipy.signal.step((num, den), T = t)
plt.plot(tout, yout)
plt.grid()
plt.title('Open Loop')
plt.ylabel('Ht(s)')
plt.xlabel('t in Seconds')
plt.show()

# part 2

# task 2

num = scipy.signal.convolve([1, 4], [1, 9])
den = scipy.signal.convolve([1, 4, 3], [2, 33, 362, 1448])
print('part 2 task 2')
print('numerator', num)
print('denominator', den)
zout, pout, kout = scipy.signal.tf2zpk(num, den)
print('z =', zout)
print('p =', pout)
print('k =', kout)

# task 4

tout, yout = scipy.signal.step((num, den), T = t)
plt.plot(tout, yout)
plt.grid()
plt.title('Closed Loop')
plt.ylabel('Ht(s)')
plt.xlabel('t in Seconds')
plt.show()