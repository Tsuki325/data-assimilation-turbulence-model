import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Set plot properties
legend_font = {
    'family': 'Times New Roman',
    'size': 18,
     'style':'italic'
}
font2 = {
    'family': 'Times New Roman',
    'style': 'italic',
    'size': 18,
}
font = {
    'family': 'Times New Roman',
    'size': 15,
}
font3 = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': 18,
}
rc = {
    "font.family": 'Times New Roman',
    "mathtext.fontset": "stix",
    "font.size": 22 
}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

fig, ax = plt.subplots()
fig.set_size_inches(8, 6) 

exp = np.loadtxt('cf.txt')
SST = np.loadtxt('wallShearStress_walls_constant1.raw', skiprows=2)
SSTNN = np.loadtxt('wallShearStress_walls_constant.raw', skiprows=2)
#SSTB = np.loadtxt('beta/wallShearStress_walls_constant.raw', skiprows=2)
#SSTP = np.loadtxt('Prt/wallShearStress_walls_constant.raw', skiprows=2)
#SSTBP = np.loadtxt('betaAndPrt/wallShearStress_walls_constant.raw', skiprows=2)

plt.plot(exp[:, 0], exp[:, 1]*0.95, label=r'Exp', color='black', marker='s', markersize=6, linestyle='', markerfacecolor='none', markeredgewidth=1.5)

# Function to plot smoothed data

plt.plot(7000000*SST[:, 0], -SST[:, 3]/0.0387/703/703/0.5, label=r'$SST$', color='red',linestyle='-')

plt.plot(7000000*SSTNN[:, 0], -SSTNN[:, 3]/0.0387/703/703/0.5, label=r'$Improved\,SST$', color='blue',linestyle='-.')

ax.minorticks_on()
ax.set_xlabel(r'$Re_{x}$')
ax.set_ylabel(r'$C_{f}$', labelpad=0, weight='bold')
import matplotlib.ticker as ticker
plt.xlim(0, 6200000)
plt.ylim(0, 0.005)
#plt.xticks(np.arange(-6, 7, 2))
#plt.ylim(0.5, 2.5)
#plt.yticks(np.arange(-0.5, 2.6, 0.5))
plt.legend(loc='upper left', ncol=1, prop=legend_font)
#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}x10^6'))
#ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[2, 3, 5, 7], numticks=6))
plt.tight_layout()
plt.show()
fig.savefig('cf.tiff',dpi=600 ,bbox_inches='tight')

