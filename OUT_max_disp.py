# this script is use to take the maxima of the dispersion curves and plot it
# -*- coding: utf-8 -*-


import numpy as np

import sqlite3
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats



#local libs
import adapt_conv_sql as acsql
from _utils import _theo_disp, format_paper, max_2D_array

format_paper()

#format_paper()                  # format for plots

sqlite3.register_adapter(np.ndarray, acsql.adapt_array)
sqlite3.register_converter("MATRIX", acsql.converter_array)


sqlite3.register_adapter(dict, acsql.adapt_dict)
sqlite3.register_converter("DICT", acsql.converter_dict)


connection = sqlite3.connect("test_FOH_RT.db",detect_types=sqlite3.PARSE_DECLTYPES)
#
cursor = connection.cursor()


maxi_l=np.empty([7, 28])
maxi_r=np.empty([7, 28])
for ii in range(1,8):
    stri="SELECT * FROM model_"+str(ii)+"_f_peak"
    cursor.execute(stri)
    data = cursor.fetchall()
    connection.commit()
    for jj in range(0,28):
        maxi_l[ii-1,jj]=data[jj][7]
        maxi_r[ii-1,jj]=data[jj][5]

connection.close()



ftfn=[0.333,0.083,0.583,0.166,0.25,0.417,0.5]



ma0_l=maxi_l[:,0]
ma1_l=maxi_l[:,1]
ma2_l=maxi_l[:,2]
ma3_l=maxi_l[:,3]
ma4_l=maxi_l[:,4]
ma5_l=maxi_l[:,5]
ma6_l=maxi_l[:,6]
ma7_l=maxi_l[:,7]
ma8_l=maxi_l[:,8]
ma9_l=maxi_l[:,9]
ma10_l=maxi_l[:,10]
ma11_l=maxi_l[:,11]
ma12_l=maxi_l[:,12]
ma13_l=maxi_l[:,13]
ma14_l=maxi_l[:,14]
ma15_l=maxi_l[:,15]
ma16_l=maxi_l[:,16]
ma17_l=maxi_l[:,17]
ma18_l=maxi_l[:,18]
ma19_l=maxi_l[:,19]
ma20_l=maxi_l[:,20]
ma21_l=maxi_l[:,21]
ma22_l=maxi_l[:,22]
ma23_l=maxi_l[:,23]
ma24_l=maxi_l[:,24]
ma25_l=maxi_l[:,25]
ma26_l=maxi_l[:,26]
ma27_l=maxi_l[:,27]


ma0_r=maxi_r[:,0]
ma1_r=maxi_r[:,1]
ma2_r=maxi_r[:,2]
ma3_r=maxi_r[:,3]
ma4_r=maxi_r[:,4]
ma5_r=maxi_r[:,5]
ma6_r=maxi_r[:,6]
ma7_r=maxi_r[:,7]
ma8_r=maxi_r[:,8]
ma9_r=maxi_r[:,9]
ma10_r=maxi_r[:,10]
ma11_r=maxi_r[:,11]
ma12_r=maxi_r[:,12]
ma13_r=maxi_r[:,13]
ma14_r=maxi_r[:,14]
ma15_r=maxi_r[:,15]
ma16_r=maxi_r[:,16]
ma17_r=maxi_r[:,17]
ma18_r=maxi_r[:,18]
ma19_r=maxi_r[:,19]
ma20_r=maxi_r[:,20]
ma21_r=maxi_r[:,21]
ma22_r=maxi_r[:,22]
ma23_r=maxi_r[:,23]
ma24_r=maxi_r[:,24]
ma25_r=maxi_r[:,25]
ma26_r=maxi_r[:,26]
ma27_r=maxi_r[:,27]



# sort to have the same arrangement like in the ftfn array

ma0_l=np.array([x for _,x in sorted(zip(ftfn,ma0_l))])
ma0_r=np.array([x for _,x in sorted(zip(ftfn,ma0_r))])

ma1_l=np.array([x for _,x in sorted(zip(ftfn,ma1_l))])
ma1_r=np.array([x for _,x in sorted(zip(ftfn,ma1_r))])

ma2_l=np.array([x for _,x in sorted(zip(ftfn,ma2_l))])
ma2_r=np.array([x for _,x in sorted(zip(ftfn,ma2_r))])

ma3_l=np.array([x for _,x in sorted(zip(ftfn,ma3_l))])
ma3_r=np.array([x for _,x in sorted(zip(ftfn,ma3_r))])

ma4_l=np.array([x for _,x in sorted(zip(ftfn,ma4_l))])
ma4_r=np.array([x for _,x in sorted(zip(ftfn,ma4_r))])

ma5_l=np.array([x for _,x in sorted(zip(ftfn,ma5_l))])
ma5_r=np.array([x for _,x in sorted(zip(ftfn,ma5_r))])

ma6_l=np.array([x for _,x in sorted(zip(ftfn,ma6_l))])
ma6_r=np.array([x for _,x in sorted(zip(ftfn,ma6_r))])

ma7_l=np.array([x for _,x in sorted(zip(ftfn,ma7_l))])
ma7_r=np.array([x for _,x in sorted(zip(ftfn,ma7_r))])

ma8_l=np.array([x for _,x in sorted(zip(ftfn,ma8_l))])
ma8_r=np.array([x for _,x in sorted(zip(ftfn,ma8_r))])

ma9_l=np.array([x for _,x in sorted(zip(ftfn,ma9_l))])
ma9_r=np.array([x for _,x in sorted(zip(ftfn,ma9_r))])

ma10_l=np.array([x for _,x in sorted(zip(ftfn,ma10_l))])
ma10_r=np.array([x for _,x in sorted(zip(ftfn,ma10_r))])

ma11_l=np.array([x for _,x in sorted(zip(ftfn,ma11_l))])
ma11_r=np.array([x for _,x in sorted(zip(ftfn,ma11_r))])

ma12_l=np.array([x for _,x in sorted(zip(ftfn,ma12_l))])
ma12_r=np.array([x for _,x in sorted(zip(ftfn,ma12_r))])

ma13_l=np.array([x for _,x in sorted(zip(ftfn,ma13_l))])
ma13_r=np.array([x for _,x in sorted(zip(ftfn,ma13_r))])

ma14_l=np.array([x for _,x in sorted(zip(ftfn,ma14_l))])
ma14_r=np.array([x for _,x in sorted(zip(ftfn,ma14_r))])

ma15_l=np.array([x for _,x in sorted(zip(ftfn,ma15_l))])
ma15_r=np.array([x for _,x in sorted(zip(ftfn,ma15_r))])

ma16_l=np.array([x for _,x in sorted(zip(ftfn,ma16_l))])
ma16_r=np.array([x for _,x in sorted(zip(ftfn,ma16_r))])

ma17_l=np.array([x for _,x in sorted(zip(ftfn,ma17_l))])
ma17_r=np.array([x for _,x in sorted(zip(ftfn,ma17_r))])

ma18_l=np.array([x for _,x in sorted(zip(ftfn,ma18_l))])
ma18_r=np.array([x for _,x in sorted(zip(ftfn,ma18_r))])

ma19_l=np.array([x for _,x in sorted(zip(ftfn,ma19_l))])
ma19_r=np.array([x for _,x in sorted(zip(ftfn,ma19_r))])

ma20_l=np.array([x for _,x in sorted(zip(ftfn,ma20_l))])
ma20_r=np.array([x for _,x in sorted(zip(ftfn,ma20_r))])

ma21_l=np.array([x for _,x in sorted(zip(ftfn,ma21_l))])
ma21_r=np.array([x for _,x in sorted(zip(ftfn,ma21_r))])

ma22_l=np.array([x for _,x in sorted(zip(ftfn,ma22_l))])
ma22_r=np.array([x for _,x in sorted(zip(ftfn,ma22_r))])

ma23_l=np.array([x for _,x in sorted(zip(ftfn,ma23_l))])
ma23_r=np.array([x for _,x in sorted(zip(ftfn,ma23_r))])

ma24_l=np.array([x for _,x in sorted(zip(ftfn,ma24_l))])
ma24_r=np.array([x for _,x in sorted(zip(ftfn,ma24_r))])

ma25_l=np.array([x for _,x in sorted(zip(ftfn,ma25_l))])
ma25_r=np.array([x for _,x in sorted(zip(ftfn,ma25_r))])

ma26_l=np.array([x for _,x in sorted(zip(ftfn,ma26_l))])
ma26_r=np.array([x for _,x in sorted(zip(ftfn,ma26_r))])

ma27_l=np.array([x for _,x in sorted(zip(ftfn,ma27_l))])
ma27_r=np.array([x for _,x in sorted(zip(ftfn,ma27_r))])




##############################################################
# plotten von Rayleigh wave max ref: disp_max_foh



#
#
# # ich bin nicht sicher ob ich vector richtig gesortet habe
# #
# fig = plt.figure(figsize=(20, 12))
#
# plt.subplot(211)
# plt.plot(np.sort(ftfn),ma0_r,'r', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma3_r,'b', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma6_r,'c', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma7_r,'y', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma9_r,'m', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma12_r,'k', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma14_r,'g', linestyle='--', marker='o',  markersize=10)
# #
# #
# #
# #
# #
# plt.plot(np.sort(ftfn),ma0_l,'r', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma3_l,'b', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma6_l,'c', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma7_l,'y', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma9_l,'m', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma12_l,'k', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma14_l,'g', linestyle='--', marker='v',  markersize=10)
#
#
#
#
# plt.legend([u'Rayleigh: 0 °',u'Rayleigh: 38 °',u'Rayleigh: 77 °',u'Rayleigh: 90 °',u'Rayleigh: 102 °',u'Rayleigh: 141 °',u'Rayleigh: 180 °',
#             u'Love: 0 °',u'Love: 38 °',u'Love: 77 °',u'Love: 90 °',u'Love: 102 °',u'Love: 141 °',u'Love: 180 °']
# , loc=2,fontsize=13, shadow=True,ncol=1)
#
# plt.ylabel('$max_{R/L} [m/s]$')
# plt.xlabel('$F_t/F_n$')
#
# plt.subplot(212)
# plt.plot(np.sort(ftfn),ma0_l/ma0_r,'r', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma3_l/ma3_r,'b', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma6_l/ma6_r,'c', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma7_l/ma7_r,'y', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma9_l/ma9_r,'m', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma12_l/ma12_r,'k', linestyle='--', marker='v',  markersize=10)
# plt.hold(True)
# plt.plot(np.sort(ftfn),ma14_l//ma14_r,'g', linestyle='--', marker='v',  markersize=10)
#
# plt.legend([u'Love/Rayleigh: 0 °',u'Love/Rayleigh: 38 °',u'Love/Rayleigh: 77 °',u'Love/Rayleigh: 90 °',u'Love/Rayleigh: 102 °',u'Love/Rayleigh: 141 °',u'Love/Rayleigh: 180 °']
# , loc=2,fontsize=13, shadow=True,ncol=1)
#
# plt.ylabel('$max_L/max_R$')
# plt.xlabel('$F_t/F_n$')
# plt.show()

#fig.savefig('max_disp.png',format='png')      # save figure







##################################################################################################
## von hier aus berechnen wir die steigung die wir for alle modelle in dependence von phi plotten wollen

slope0_r,intercept,r_value,p_value,std_err0_r = stats.linregress(ftfn, ma0_r)
slope0_l,intercept,r_value,p_value,std_err0_l = stats.linregress(ftfn, ma0_l)
slope0_lr,intercept,r_value,p_value,std_err0_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma0_l,ma0_r)])


slope1_r,intercept,r_value,p_value,std_err1_r = stats.linregress(ftfn, ma1_r)
slope1_l,intercept,r_value,p_value,std_err1_l = stats.linregress(ftfn, ma1_l)
slope1_lr,intercept,r_value,p_value,std_err1_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma1_l,ma1_r)])



slope2_r,intercept,r_value,p_value,std_err2_r = stats.linregress(ftfn, ma2_r)
slope2_l,intercept,r_value,p_value,std_err2_l = stats.linregress(ftfn, ma2_l)
slope2_lr,intercept,r_value,p_value,std_err2_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma2_l,ma2_r)])



slope3_r,intercept,r_value,p_value,std_err3_r = stats.linregress(ftfn, ma3_r)
slope3_l,intercept,r_value,p_value,std_err3_l = stats.linregress(ftfn, ma3_l)
slope3_lr,intercept,r_value,p_value,std_err3_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma3_l,ma3_r)])

slope4_r,intercept,r_value,p_value,std_err4_r = stats.linregress(ftfn, ma4_r)
slope4_l,intercept,r_value,p_value,std_err4_l = stats.linregress(ftfn, ma4_l)
slope4_lr,intercept,r_value,p_value,std_err4_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma4_l,ma4_r)])


slope5_r,intercept,r_value,p_value,std_err5_r = stats.linregress(ftfn, ma5_r)
slope5_l,intercept,r_value,p_value,std_err5_l = stats.linregress(ftfn, ma5_l)
slope5_lr,intercept,r_value,p_value,std_err5_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma5_l,ma5_r)])



slope6_r,intercept,r_value,p_value,std_err6_r = stats.linregress(ftfn, ma6_r)
slope6_l,intercept,r_value,p_value,std_err6_l = stats.linregress(ftfn, ma6_l)
slope6_lr,intercept,r_value,p_value,std_err6_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma6_l,ma6_r)])


slope7_r,intercept,r_value,p_value,std_err7_r = stats.linregress(ftfn, ma7_r)
slope7_l,intercept,r_value,p_value,std_err7_l = stats.linregress(ftfn, ma7_l)
slope7_lr,intercept,r_value,p_value,std_err7_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma7_l,ma7_r)])



slope8_r,intercept,r_value,p_value,std_err8_r = stats.linregress(ftfn, ma8_r)
slope8_l,intercept,r_value,p_value,std_err8_l = stats.linregress(ftfn, ma8_l)
slope8_lr,intercept,r_value,p_value,std_err8_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma8_l,ma8_r)])



slope9_r,intercept,r_value,p_value,std_err9_r = stats.linregress(ftfn, ma9_r)
slope9_l,intercept,r_value,p_value,std_err9_l = stats.linregress(ftfn, ma9_l)
slope9_lr,intercept,r_value,p_value,std_err9_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma9_l,ma9_r)])



slope10_r,intercept,r_value,p_value,std_err10_r = stats.linregress(ftfn, ma10_r)
slope10_l,intercept,r_value,p_value,std_err10_l = stats.linregress(ftfn, ma10_l)
slope10_lr,intercept,r_value,p_value,std_err10_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma10_l,ma10_r)])


slope11_r,intercept,r_value,p_value,std_err11_r = stats.linregress(ftfn, ma11_r)
slope11_l,intercept,r_value,p_value,std_err11_l = stats.linregress(ftfn, ma11_l)
slope11_lr,intercept,r_value,p_value,std_err11_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma11_l,ma11_r)])


slope12_r,intercept,r_value,p_value,std_err12_r = stats.linregress(ftfn, ma12_r)
slope12_l,intercept,r_value,p_value,std_err12_l = stats.linregress(ftfn, ma12_l)
slope12_lr,intercept,r_value,p_value,std_err12_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma12_l,ma12_r)])


slope13_r,intercept,r_value,p_value,std_err13_r = stats.linregress(ftfn, ma13_r)
slope13_l,intercept,r_value,p_value,std_err13_l = stats.linregress(ftfn, ma13_l)
slope13_lr,intercept,r_value,p_value,std_err13_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma13_l,ma13_r)])


slope14_r,intercept,r_value,p_value,std_err14_r = stats.linregress(ftfn, ma14_r)
slope14_l,intercept,r_value,p_value,std_err14_l = stats.linregress(ftfn, ma14_l)
slope14_lr,intercept,r_value,p_value,std_err14_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma14_l,ma14_r)])


slope15_r,intercept,r_value,p_value,std_err15_r = stats.linregress(ftfn, ma15_r)
slope15_l,intercept,r_value,p_value,std_err15_l = stats.linregress(ftfn, ma15_l)
slope15_lr,intercept,r_value,p_value,std_err15_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma15_l,ma15_r)])


slope16_r,intercept,r_value,p_value,std_err16_r = stats.linregress(ftfn, ma16_r)
slope16_l,intercept,r_value,p_value,std_err16_l = stats.linregress(ftfn, ma16_l)
slope16_lr,intercept,r_value,p_value,std_err16_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma16_l,ma16_r)])



slope17_r,intercept,r_value,p_value,std_err17_r = stats.linregress(ftfn, ma17_r)
slope17_l,intercept,r_value,p_value,std_err17_l = stats.linregress(ftfn, ma17_l)
slope17_lr,intercept,r_value,p_value,std_err17_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma17_l,ma17_r)])


slope18_r,intercept,r_value,p_value,std_err18_r = stats.linregress(ftfn, ma18_r)
slope18_l,intercept,r_value,p_value,std_err18_l = stats.linregress(ftfn, ma18_l)
slope18_lr,intercept,r_value,p_value,std_err18_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma18_l,ma18_r)])


slope19_r,intercept,r_value,p_value,std_err19_r = stats.linregress(ftfn, ma19_r)
slope19_l,intercept,r_value,p_value,std_err19_l = stats.linregress(ftfn, ma19_l)
slope19_lr,intercept,r_value,p_value,std_err19_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma19_l,ma19_r)])


slope20_r,intercept,r_value,p_value,std_err20_r = stats.linregress(ftfn, ma20_r)
slope20_l,intercept,r_value,p_value,std_err20_l = stats.linregress(ftfn, ma20_l)
slope20_lr,intercept,r_value,p_value,std_err20_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma20_l,ma20_r)])


slope21_r,intercept,r_value,p_value,std_err21_r = stats.linregress(ftfn, ma21_r)
slope21_l,intercept,r_value,p_value,std_err21_l = stats.linregress(ftfn, ma21_l)
slope21_lr,intercept,r_value,p_value,std_err21_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma21_l,ma21_r)])


slope22_r,intercept,r_value,p_value,std_err22_r = stats.linregress(ftfn, ma22_r)
slope22_l,intercept,r_value,p_value,std_err22_l = stats.linregress(ftfn, ma22_l)
slope22_lr,intercept,r_value,p_value,std_err22_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma22_l,ma22_r)])


slope23_r,intercept,r_value,p_value,std_err23_r = stats.linregress(ftfn, ma23_r)
slope23_l,intercept,r_value,p_value,std_err23_l = stats.linregress(ftfn, ma23_l)
slope23_lr,intercept,r_value,p_value,std_err23_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma23_l,ma23_r)])


slope24_r,intercept,r_value,p_value,std_err24_r = stats.linregress(ftfn, ma24_r)
slope24_l,intercept,r_value,p_value,std_err24_l = stats.linregress(ftfn, ma24_l)
slope24_lr,intercept,r_value,p_value,std_err24_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma24_l,ma24_r)])


slope25_r,intercept,r_value,p_value,std_err25_r = stats.linregress(ftfn, ma25_r)
slope25_l,intercept,r_value,p_value,std_err25_l = stats.linregress(ftfn, ma25_l)
slope25_lr,intercept,r_value,p_value,std_err25_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma25_l,ma25_r)])


slope26_r,intercept,r_value,p_value,std_err26_r = stats.linregress(ftfn, ma26_r)
slope26_l,intercept,r_value,p_value,std_err26_l = stats.linregress(ftfn, ma26_l)
slope26_lr,intercept,r_value,p_value,std_err26_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma26_l,ma26_r)])


slope27_r,intercept,r_value,p_value,std_err27_r = stats.linregress(ftfn, ma27_r)
slope27_l,intercept,r_value,p_value,std_err27_l = stats.linregress(ftfn, ma27_l)
slope27_lr,intercept,r_value,p_value,std_err27_lr = stats.linregress(ftfn, [x/y for x,y in zip(ma27_l,ma27_r)])

slope_r=[slope0_r,slope1_r,slope2_r,slope3_r,slope4_r,slope5_r,slope6_r,slope7_r,slope8_r,slope9_r,slope10_r,slope11_r,slope12_r
    , slope13_r, slope14_r, slope15_r, slope16_r, slope17_r, slope18_r, slope19_r, slope20_r, slope21_r
    , slope22_r, slope23_r, slope24_r, slope25_r, slope26_r, slope27_r]

err_r=[std_err0_r,std_err1_r,std_err2_r,std_err3_r,std_err4_r,std_err5_r,std_err6_r,std_err7_r,std_err8_r,std_err9_r,std_err10_r,std_err11_r
       ,std_err12_r,std_err13_r,std_err14_r,std_err15_r,std_err16_r,std_err17_r,std_err18_r,std_err19_r,std_err20_r,std_err21_r,
       std_err22_r, std_err23_r, std_err24_r, std_err25_r, std_err26_r, std_err27_r ]



slope_l=[slope0_l,slope1_l,slope2_l,slope3_l,slope4_l,slope5_l,slope6_l,slope7_l,slope8_l,slope9_l,slope10_l,slope11_l,slope12_l
    , slope13_l, slope14_l, slope15_l, slope16_l, slope17_l, slope18_l, slope19_l, slope20_l, slope21_l
    , slope22_l, slope23_l, slope24_l, slope25_l, slope26_l, slope27_l]


err_l=[std_err0_l,std_err1_l,std_err2_l,std_err3_l,std_err4_l,std_err5_l,std_err6_l,std_err7_l,std_err8_l,std_err9_l,std_err10_l,std_err11_l
       ,std_err12_l,std_err13_l,std_err14_l,std_err15_l,std_err16_l,std_err17_l,std_err18_l,std_err19_l,std_err20_l,std_err21_l,
       std_err22_l, std_err23_l, std_err24_l, std_err25_l, std_err26_l, std_err27_l ]


slope_lr=[slope0_lr,slope1_lr,slope2_lr,slope3_lr,slope4_lr,slope5_lr,slope6_lr,slope7_lr,slope8_lr,slope9_lr,slope10_lr,slope11_lr,slope12_lr
    , slope13_lr, slope14_lr, slope15_lr, slope16_lr, slope17_lr, slope18_lr, slope19_lr, slope20_lr, slope21_lr
    , slope22_lr, slope23_lr, slope24_lr, slope25_lr, slope26_lr, slope27_lr]
slope_lr=np.append(slope_lr,slope_lr[0])

err_lr=[std_err0_lr,std_err1_lr,std_err2_lr,std_err3_lr,std_err4_lr,std_err5_lr,std_err6_lr,std_err7_lr,std_err8_lr,std_err9_lr,std_err10_lr,std_err11_lr
       ,std_err12_lr,std_err13_lr,std_err14_lr,std_err15_lr,std_err16_lr,std_err17_lr,std_err18_lr,std_err19_lr,std_err20_lr,std_err21_lr,
       std_err22_lr, std_err23_lr, std_err24_lr, std_err25_lr, std_err26_lr, std_err27_lr ]
err_lr=np.append(err_lr,err_lr[0])


phi = np.linspace(0.0, 2.0 * np.pi * (28 - 1) / 28.0, 28)*360/(2*np.pi)
phi=np.append(phi,360)
plt.subplot(121)
# plt.plot(phi,map(abs,slope_r),'g', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
# plt.plot(phi,slope_l,'b', linestyle='--', marker='o',  markersize=10)
# plt.hold(True)
plt.plot(phi,slope_lr,'g', linestyle='--', marker='o',  markersize=10)
plt.errorbar(phi, slope_lr, err_lr,color='g', linestyle='None', marker='^')
# plt.errorbar(phi, slope_l, err_l,color='b', linestyle='None', marker='^')

plt.xlabel(u'degree °')
plt.ylabel('slope')
plt.legend([u'Love/Rayleigh']
, loc=2,fontsize=13, shadow=True,ncol=1)

phi = np.linspace(0.0, 2.0 * np.pi * (28 - 1) / 28.0, 28)
phi=np.append(phi,2*np.pi)
#actual plotting
ax = plt.subplot(122, projection='polar')
ax.plot(phi, slope_lr,'g', linestyle='--', marker='o',  markersize=10)

# ctf = ax.contourf(phi, slope_lr)
# cb=plt.colorbar(ctf)
# ax.grid(True)
# plt.rcParams.update({'font.size': 16})          # macht alle fonts grosser
# ax.fontcolor='w'

plt.show()
#
