import os
import sys
import re
import numpy as np
import pandas as pd
import subprocess
import pandas as pd

import numpy as np
import numpy.fft as fft
from numpy.fft import fftfreq

import scipy

import scipy.io
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree

import matplotlib.pyplot as plt

import scipy.integrate
from scipy.integrate.quadpack import quad

# print(help(quad))
scipy.special.b
[a, b] = [0, 1]


def f1(x): return np.exp(-x**2)


def kernel(x): return np.exp(-x**2)


quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08,
     limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50)


[xmin,xmax,nx]=[-2,2,100]

xx = np.linspace(xmin, xmax, nx)
yy = np.heaviside(xx, 0)
yint=np.trapz(yy,xx)
yy/=yint
ylist=[yy.tolist()]
ylist
xlist=[xx.tolist()]

# x2 = np.convolve(xx[::],xx[::])
nbox=4
for i in range(nbox):
    y2 = np.convolve(ylist[-1],yy[::])
    y2=y2[::2]

    y2int = np.trapz(y2,xx)
    y2/=y2int 
    xlist.append(xx.tolist())
    ylist.append(y2.tolist())

plt.figure()
for ii,(x0,y0) in enumerate(zip(xlist,ylist)):
    1;
    plt.plot(x0,y0,'-o',alpha=0.25,label=str(ii))
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Iterated Convolution')
plt.show()
