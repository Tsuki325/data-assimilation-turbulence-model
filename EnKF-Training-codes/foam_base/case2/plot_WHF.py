
import numpy as np
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pyplot as plt
import pdb
# set plot properties
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    # 'image.cmap': 'viridis',
    'axes.grid': False,
    'savefig.dpi': 300,
    'axes.labelsize': 20, # 10
    'axes.titlesize': 20,
    'font.size': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex': True,
    'figure.figsize': [5, 4],
    'font.family': 'serif',
}


exp = np.loadtxt('34_hf.txt')
SST = np.loadtxt('wallHeatFlux_walls_constant1.raw',skiprows=2)
SSTNN = np.loadtxt('wallHeatFlux_walls_constant.raw',skiprows=2)
fig, ax = plt.subplots()
a=100
b=220
plt.plot(exp[:, 0]/1.3, exp[:, 1], 'ko', markersize=7, fillstyle='none', label='exp')
plt.plot(SST[:,0],-SST[:,3]/61000, 'r-', lw=1, label='SST') #, markersize=2)
print(SST[:,3])
plt.plot(SSTNN[:,0],-SSTNN[:,3]/61000, 'g-', lw=1, label='SSTNN') #, markersize=2)
'''
interp_func = interpolate.interp1d(SST[:,0],-SST[:,3]/61000)
xc = exp[:,0]/1.3
exp_interplot = interp_func(xc)
EXPToMesh=np.vstack([xc,exp_interplot]).T

plt.plot(xc, exp_interplot ,'gs', markersize=7, fillstyle='none', label='expToMesh')

'''

ax.set_xlabel(f'$x/δ$')
ax.set_ylabel(f'HEAT FLUX')
plt.xlim(-0.06,0.06)
plt.ylim(0,50)
plt.legend()
plt.tight_layout()
plt.savefig('c_ramp_HF.png')
plt.show()




