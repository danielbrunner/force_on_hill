import numpy as np

import sqlite3
import matplotlib.pyplot as plt


#local libs
import adapt_conv_sql as acsql
from _utils import _theo_disp, format_paper, max_2D_array

#format_paper()                  # format for plots

sqlite3.register_adapter(np.ndarray, acsql.adapt_array)
sqlite3.register_converter("MATRIX", acsql.converter_array)


sqlite3.register_adapter(dict, acsql.adapt_dict)
sqlite3.register_converter("DICT", acsql.converter_dict)


connection = sqlite3.connect("test_FOH_RT.db",detect_types=sqlite3.PARSE_DECLTYPES)
#
cursor = connection.cursor()


cursor.execute("SELECT * FROM RT_sqrt")
data = cursor.fetchall()

connection.commit()

connection.close()


# # calculate the maximume values from R and T
# max_T=np.zeros((75,29))
# max_R=np.zeros((75,29))
# for ii in range(0,74):
#     for jj in range(0,28):
#         max_R[ii,jj]=max(data[jj][2][:,ii]**2)
#         max_T[ii,jj]=max(data[jj][3][:,ii]**2)
#
# #max_R=np.concatenate([max_R,max_R[:,0:]], axis=1)
# max_R[:,-1]=max_R[:,0]
# max_T[:,-1]=max_T[:,0]
#
#
# azimuth = np.radians(np.linspace(0, 360, 29))
# #azimuth=np.append(azimuth,azimuth[0])
# radius = np.linspace(0, 24*400, num=75)
#
#
# #actual plotting
#
# # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(20, 12))
# # ctf = ax.contourf(azimuth,radius, max_T/max_R)
# # cb=plt.colorbar(ctf)
# # ax.grid(True)
# # plt.rcParams.update({'font.size': 16})          # macht alle fonts grosser
# #
# # #fig.savefig('model_7_polar'+'.png',format='png')      # save figure
# #
# #
# # plt.show()




####################################################### formula for RT_sqrt table in DB FOH_RT.db
# plot maximal of all radial/transversal versus F_t/F_n

# TR_sqrt=max_T/max_R
#
#
# max_TR_sqrt=max_2D_array(TR_sqrt)
#
# print(max_TR_sqrt)

ftfn=np.sort(data[0][0:])
ftfn=np.array([ftfn[0],ftfn[4],ftfn[6]])
x_ftfn=np.array([0.083,0.333,0.583])
print(x_ftfn)
print(ftfn)
print(ftfn/x_ftfn)

plt.plot(x_ftfn,ftfn)
plt.show()
change=1

###############
# werte fur sql
# 1. WERT: WINKEL PHI
# 2- WERT: phi, meta, data_R, data_T, disp_R, max_R, disp_T, max_T
# 3. 1. SEISMOGRAM
# 3. .2 RADIUS FROM SOURCE