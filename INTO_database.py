# class to change data from 2D to 3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import inspect
import os.path
import re
from scipy import signal

import sqlite3
import json

# local libs
import adapt_conv_sql as acsql
from lib_data_processing import data_processing
from _utils import _theo_disp

## ##############Gebrauchsanweisung:
# model 1 muss fuer file aber auch fuer table veraendert werden



################## DATA AND PARAMETER

data_str='/home/djamel/PHD_projects/force_on_hill/results_seismogram/model_7_f_peak.npy'
data = np.load(data_str)

dt=0.05
t_n = 700
n_circ = 28
n_r = 75

dist=500                # distance receiver
ttn=0                   # model 1=0.333; model 2=0.083 model 3=0.583        -> F_t/F_n
ltd=[]                  # lambda/hill length
dist_m=20               # distance microarray



data=data_processing(data,dt,t_n,n_circ,n_r)
data.radial_transversal()
# #data.rotation(dist_m)
data.dispersion(100,4000,1500,dist)

sqlite3.register_adapter(np.ndarray, acsql.adapt_array)
sqlite3.register_converter("MATRIX", acsql.converter_array)


sqlite3.register_adapter(dict, acsql.adapt_dict)
sqlite3.register_converter("DICT", acsql.converter_dict)



# # # gr: gradient of the hill
# # # nu_t: number of timesamples

meta = {'dt': dt, 'n_circ': n_circ, 'n_r': n_r,'n_t':t_n,'gra':0.0,'dist':dist,'ttn':ttn}
#
#
# #
# # ##### sql part #############
# #
connection = sqlite3.connect("test.db",detect_types=sqlite3.PARSE_DECLTYPES)

cursor = connection.cursor()


################## save in sql
cursor.execute("CREATE TABLE model_7_f_peak (phi INT, meta DICT, data_R MATRIX, data_T MATRIX"
               ", disp_R MATRIX, max_R float, disp_T MATRIX, max_T float)")
for ii in range(0,len(data.phii[:])):
    cursor.execute("INSERT INTO model_7_f_peak VALUES (?,?,?,?,?,?,?,?)", (data.phii[ii],meta,data.data_R[:,:,ii],data.data_T[:,:,ii]
                                                        ,data.DISP_R[:,:,ii],data.max_R[ii,0],data.DISP_T[:,:,ii],data.max_T[ii,0]))
connection.commit()
connection.close()
#####################


##############################################
# cursor.execute("SELECT * FROM model_1_f_peak")
# data = cursor.fetchall()
# connection.commit()
# connection.close()
#################

#
# # print(data[0][1])
# #
#
# #
# d_a=7500   #distance first receiver
# d_r=500    # distance receivers
# num_r=75    # number of receivers
#
# n_circular_rec=28                            # number of receiver in circular direction
# n_radial_rec=75                             # numuber of radial receivers
# phii=np.arange(0,360,360/float(n_circular_rec),dtype=float)
#
#
# #
# # print(data[0][2][:,0])
#
# ampl = 1000  # amplitude vergroesserung
# #
# #
# #
# for kk in range(5,6):
#
#     ### wiggle output
#     fig=plt.figure(figsize=(15, 16))
#     hh = 0  # used to move the seismogram uppwards to build the wiggles
#     for ii in range(0,75):
#         if ii % 4 == 0:
#
#             plt.plot( d_a + ii * d_r + ampl * data[0][2][:,ii]/max(data[0][2][:,ii]) , 'g', linewidth=2)
#         hh = hh + 1
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # plt.plot(data.data_R[:,10,10],'r')
# # plt.hold
# # plt.plot(data.data_T[:,10,15],'g')
# # plt.show()
#
# #data.disperison(100,4000,1500,dist)
#
#
# swi=0
# freq = np.fft.fftfreq(len(dataa.data_R[:, 0, 0]), dataa.dt)
# fig = plt.figure(figsize=(9, 10))
# _theo_disp(swi)
# plt.imshow(data[7][3][:,0:200]*data[7][4], extent=(freq[0], freq[200], 1500, 4000), aspect='auto',cmap='gnuplot')
# plt.colorbar()
# plt.xlabel('frequency [Hz]', fontsize=17)
# plt.ylabel('velocity [m/s]', fontsize=17)
# plt.show()