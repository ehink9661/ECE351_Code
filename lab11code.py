# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Ethan Hinkle                                                 #
# ECE 351-51                                                   #
# Lab 11                                                       #
# April 5th, 2022                                              #
# This lab revlves around using z transformations in python.   #
# The zplane function was provided fir us in this lab. We      #
# then use python to plot and analyze these functions.         #
#                                                              #
################################################################
"""
Created on Tue Apr  5 19:42:32 2022

@author: Anaconda
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from matplotlib import patches    

# task 3

num = [2, -40]
den = [1, -10, 16]
rout, pout, kout = scipy.signal.residue(num, den)
print('task 3')
print('r =', rout)
print('p =', pout)
print('k =', kout)


# task 4

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    
    # get a figure/plot
    ax = plt.subplot(111)
    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)
    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1
    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)
    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()
    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k

zplane(num, den)

# task 5

w, h = scipy.signal.freqz(num, den, whole = True)

plt.figure(figsize = (10, 7))
plt.subplot (2,1,1)
plt.plot(w, abs(h))
plt.grid()
plt.title('Task 5')
plt.ylabel('magnitude')

plt.subplot (2,1,2)
plt.plot(w, np.angle(h, deg = True))
plt.grid()
plt.xlabel('radians (w)')
plt.ylabel('angle (deg)')