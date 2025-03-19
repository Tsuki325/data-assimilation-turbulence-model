
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


exp = np.loadtxt('30du_heatflux.txt')
SST = np.loadtxt('wallHeatFlux_walls_constant.raw',skiprows=2)
#SST = np.loadtxt('wallHeatFlux_walls_constantSST.raw',skiprows=2)
fig, ax = plt.subplots()

plt.plot(exp[:, 0], exp[:, 1], 'ko', markersize=7, fillstyle='none', label='exp')
#plt.plot(SST[:, 0], -SST[:, 3]/67000, 'r-', lw=1, label='SST') #, markersize=2)
print(SST[:, 3])


plt.plot(SST[:,0], -SST[:, 3]/67000, 'g-', lw=3, label='SST') #, markersize=2)
#plt.plot(SSTNN[90:155,0], -SSTNN[90:155, 3]/67000, 'r-', lw=3, label='SST_NN') #, markersize=2)

ax.set_xlabel(f'$x/δ$')
ax.set_ylabel(f'HEAT FLUX')
plt.xlim(-0.1,0.1)
plt.legend()
plt.tight_layout()
plt.savefig('c_ramp_HF.png')
plt.show()




