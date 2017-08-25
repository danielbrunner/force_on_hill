# class to change data from 2D to 3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import inspect
import os.path
import re
from scipy import signal



data_str='/home/djamel/PHD_projects/force_on_hill/results_seismogram/model_1_f_peak.npy'
data = np.load(data_str)

dt=0.01
f_n=1/(dt*2)

f=0.01
fe=12
Wn=f/f_n
Wn = [f / f_n, fe/f_n]




t_n = 700
n_circ = 63
n_r = 28

# brauche ich fuer jeden vierten output
data_t = np.zeros((t_n))
data_X = np.zeros((t_n, n_r, n_circ))
data_Y = np.zeros((t_n, n_r, n_circ))
data_Z = np.zeros((t_n, n_r, n_circ))
rot_X = np.zeros((t_n, n_r, n_circ))
rot_Y = np.zeros((t_n, n_r, n_circ))
rot_Z = np.zeros((t_n, n_r, n_circ))
data_t[:] = data[:, 0, 0]
ll = 0
for ii in range(0, n_circ):
    for jj in range(0, n_r):

        data_X[:, jj, ii] = data[:, 1, ll]
        data_Y[:, jj, ii] = data[:, 2, ll]
        data_Z[:, jj, ii] = data[:, 3, ll]
        rot_X[:,jj,ii]=data[:, 4, ll]
        rot_Y[:,jj,ii]=data[:, 5, ll]
        rot_Z[:,jj,ii]=data[:, 6, ll]
        ll = ll + 1


