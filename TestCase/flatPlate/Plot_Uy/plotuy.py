import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
# set plot properties 
font2 = {'family' : 'Times New Roman',
    'style': 'italic',
    #'weight':'bold',
'size' :18,
}
font = {'family' : 'Times New Roman',
    #'weight':'bold',
'size' : 15,
}
font3 = {'family' : 'Times New Roman',
    'weight':'bold',
'size' : 18,
}
legend_font = {
  
    'family': 'Times New Roman',  
    'size': 17,
    'weight':'bold' 
}

rc = {
    "font.family": 'Times New Roman',
    "mathtext.fontset": "stix",
    "font.size": 22 
}
plt.rcParams.update(rc)

plt.rcParams["font.serif"] = ["Times New Roman"]+ plt.rcParams["font.serif"]
fig, ax = plt.subplots()
fig.set_size_inches(8, 6) 
#tw=np.genfromtxt('wa
#p=np.genfromtxt('p_walls_constant.raw',comments='#')
#nu_rho=np.genfromtxt('line_nu_rho.xy')
Usst = np.loadtxt('sst/line_U.xy')
Unn = np.loadtxt('nn/line_U.xy')
exp = np.loadtxt('exp5')
exp1 = np.loadtxt('exp1')
exp2 = np.loadtxt('exp2')
exp3 = np.loadtxt('exp3')
exp4 = np.loadtxt('exp4')
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#f_linear_tw = interp1d(tw[:,0], tw[:,3], kind='linear')
#tw_ = f_linear_tw(0.2)

#yplus = U[:,0]*np.sqrt(-1*tw_/nu_rho[:,2])/nu_rho[:,1]
#Uplus = U[:,1]/np.sqrt(-1*tw_/nu_rho[:,2])
plt.plot(Usst[:,0],Usst[:,1]/703.360, '#d62728', lw=1.5, label='$SST$',linestyle='-')

plt.plot(Unn[:,0],Unn[:,1]/703.360, 'blue', lw=1.5, label='$Improved\,SST$',linestyle='-.')
plt.plot(exp[:,1]*0.3048,0.3048*exp[:,2]/703.360,label=r'$Exp$' ,color='black',marker='+',markersize=6,linestyle='',markerfacecolor='none',markeredgewidth=1)
#plt.plot(yplus,yplus)
plt.plot(exp1[:,1]*0.3048,0.3048*exp1[:,2]/703.360,label=r'$Coles 23: Fence Trip$' ,color='grey',marker='x',markersize=6,linestyle='',markerfacecolor='none',markeredgewidth=1)
plt.plot(exp2[:,1]*0.3048,0.3048*exp2[:,2]/703.360,label=r'$Coles 22: Fence Trip$' ,color='black',marker='o',markersize=6,linestyle='',markerfacecolor='none',markeredgewidth=0.75)
plt.xscale('log')


# 设置次刻度
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))

ax.set_xlabel(r'$y(m)$')
ax.set_ylabel(r'$u/u_{ref}$',labelpad=1,weight='bold')

plt.ylim(0,1.03)
plt.xlim(1e-5,0.1)
plt.legend(loc='upper left', ncol=1, prop=legend_font)
#plt.ylim(0,40)
# Show the plot
plt.savefig('y_u.tiff',dpi=600,bbox_inches='tight')
plt.show()
