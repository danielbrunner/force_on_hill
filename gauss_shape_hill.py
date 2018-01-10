# this script is used to create a hill with a gauss shape and the layers

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#local libs
from _utils import _theo_disp,format_paper

############################ read eigenmodes

format_paper()

#### read depth values and calculate the depth vector

temp=pd.read_csv("./theo_EM/SLDER_1.0_Hz.TXT", skiprows=range(0, 3), nrows=66, delim_whitespace=True)
d=np.array(temp)
dd=np.cumsum(d[:,1])*1000-100     # da wir nur distance haen macht diese fkt. eine plot array


# read eigenmodes values
temp=pd.read_csv("./theo_EM/SLDER_1.0_Hz.TXT", skiprows=range(0, 75), delim_whitespace=True)
prov=np.array(temp)
UT=prov[:,1]

temp=pd.read_csv("./theo_EM/SRDER_1.0_Hz.TXT", skiprows=range(0, 75), delim_whitespace=True)
prov=np.array(temp)
UR=prov[:,1]
UZ=prov[:,3]

# read elipicity data
temp = pd.read_csv("/home/djamel/PHD_projects/scatering_Paper/theo_EM_scattering/SREGN_1.0_Hz.TXT",
                   skiprows=range(0, 3), delim_whitespace=True)
global c_r
c_r = np.array(temp)[0, 3] * 1000  # read group velocity of Rayleigh waves


# plt.plot(d,UR,d,UT)
# plt.show()



############################# Parameter
size = 100000                                          # lange des models
number_grid=3000                                       # gitterpunkte in eine richtung
sigma_x = 1050.                                     # je grosser sigma umso breiter gauss kurve
mu=size/2                                            # verschiebung der gauss kurve in eine richtung -> ich lasse kurve immer in der mitte
h=6000000000.                                          # hohe hill

freq=1.0

l_RW=c_r/freq                                               # wavelength at certain frequency Rayleigh wave

###########################

xx = np.linspace(0, size, number_grid)
yy = np.linspace(0, size, number_grid)

x, y = np.meshgrid(xx, yy)
z = h*(1/(2*np.pi*sigma_x*sigma_x) * np.exp(-((x-mu)**2/(2*(sigma_x)**2)
     + (y-mu)**2/(2*sigma_x**2))))



########## plottet bei der Halfte des models um ein querschnitt der gausskurve zu bekommen und die steigung zu berchnen
pp=z[number_grid/2,:]


dx = np.diff(xx)
d=5
pd=np.gradient(pp,dx[0])      # ableitung der gauss kurve und die steigung zu berechnen



################# findet max index und maximum fur den gradienten
maximum=0
for i,value in enumerate(pd):
    if value>maximum:
        maximum=value
        index=i

################# findet max index und maximum

maximum_m=0
for i,value in enumerate(pp):
    if value>maximum_m:
        maximum_m=value
        index_m=i


################ berechnet die lange der gaussglocke am Boden
dumb_minn=10000000000
dumb_maxx=0
for ii in range(0,len(pp[:])-1):
    if pp[ii]>0.0001:
        dumb_min = ii
        dumb_max = ii
        if dumb_minn >= ii:
            dumb_minn=dumb_min
        if dumb_maxx <= ii:
            dumb_maxx = dumb_max
len_gauss=[dumb_minn,dumb_maxx]
len_gauss=[ii *size/number_grid for ii in len_gauss]

pos_len_plt=(len_gauss[1]+len_gauss[0])/2               #brauche ich um mitte zu finden fur lange der gausskurve im plot
len_gauss_plt=(len_gauss[1]-len_gauss[0])                       #lange der gausskurve


#################################################################

print('die maximale steigung ist:'+'\033[91m'+'\033[1m' +str(maximum)+ '\033[0m'+'\n')
print('maximaler punkt ist am Ort: x='+'\033[91m'+'\033[1m'+str(size/2)+' y='+str(index*size/number_grid)+' z='+str(z[number_grid/2,index]) +'\033[0m')


######### plot gauss querschnitt

plt.plot(x[1,:],pp)
plt.hold(True)
plt.plot(x[1,0:],pd)
plt.hold(True)
plt.plot(index*size/number_grid,z[number_grid/2,index], 'ro')
plt.hold(True)
plt.plot(index_m*size/number_grid,max(pp), 'go')
plt.plot(len_gauss,[0,0])
plt.hold(True)
plt.plot([0, 100000],[-1000,-1000])
plt.hold(True)
plt.plot([0, 100000],[-2000,-2000])
plt.hold(True)
plt.plot([0, 100000],[-4000,-4000])
plt.hold(True)
plt.plot([0, 100000],[-7000,-7000])
plt.hold(True)
plt.plot(UT*3000+50000,-dd)

plt.xlim((40000,60000))
plt.ylim((0,1500))

plt.xlabel('length [m]')
plt.ylabel('heigth [m]')
plt.annotate('maximum gradeint: '+str(round(maximum,3)), xy=(index*size/number_grid-size/6, z[number_grid/2,index]))
#plt.annotate('maximaler punkt ist am Ort: x='+str(int(size/2))+' y='+str(int(index*size/number_grid))+' z='+str(int(z[number_grid/2,index])), xy=(index*size/number_grid-size/6, z[number_grid/2,index]-z[number_grid/2,index]/20))
plt.annotate('maximum value: '+str(int(max(pp))), xy=(index_m*size/number_grid,max(pp)+max(pp)/60))
plt.annotate('length gauss curve: '+str(len_gauss_plt)+' m', xy=(pos_len_plt,0-max(pp)/30))
plt.annotate('length fundamental Rayleigh mode/length gauss curve: '+str(len_gauss_plt/l_RW)+' m', xy=(pos_len_plt,0-max(pp)/15))

plt.show()


# ########## plotting
# plt.contourf(x, y, z, cmap='Blues')
# cbar=plt.colorbar()
# cbar.set_label('height [m]')
#
# plt.xlabel('length [m]')
# plt.ylabel('length [m]')
# plt.show()
