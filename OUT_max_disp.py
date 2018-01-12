# this script is use to take the maxima of the dispersion curves and plot it

import numpy as np

import sqlite3
import matplotlib.pyplot as plt
import matplotlib as mpl



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


maxi=np.empty([8, 14])
for ii in range(1,8):
    stri="SELECT * FROM model_"+str(ii)+"_f_peak"
    cursor.execute(stri)
    data = cursor.fetchall()
    connection.commit()
    for jj in range(0,14):
        maxi[ii,jj]=data[jj][7]

connection.close()

print(maxi)