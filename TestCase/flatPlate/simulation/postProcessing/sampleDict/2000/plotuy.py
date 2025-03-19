import numpy as np
#tw=np.genfromtxt('wallShearStress_walls_constant.raw',comments='#')
#p=np.genfromtxt('p_walls_constant.raw',comments='#')
#nu_rho=np.genfromtxt('line_nu_rho.xy')
U = np.loadtxt('line_U.xy')
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#f_linear_tw = interp1d(tw[:,0], tw[:,3], kind='linear')
#tw_ = f_linear_tw(0.2)


#yplus = U[:,0]*np.sqrt(-1*tw_/nu_rho[:,2])/nu_rho[:,1]
#Uplus = U[:,1]/np.sqrt(-1*tw_/nu_rho[:,2])
plt.plot(U[:,0],U[:,1])
#plt.plot(yplus,yplus)
#plt.plot(yplus,5 + 2.439* np.log(yplus))
plt.xscale('log')
#plt.ylim(0,40)
# Show the plot
plt.show()
