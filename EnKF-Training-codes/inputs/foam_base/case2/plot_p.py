
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


exp = np.loadtxt('34_p.txt')
SST15 = np.loadtxt('p_walls_constant.raw',skiprows=2)
SST = np.loadtxt('p_walls_constant1.raw',skiprows=2)

fig, ax = plt.subplots()
a=100
b=140
plt.plot(exp[:, 0]/0.9, exp[:, 1], 'ko', markersize=7, fillstyle='none', label='exp')
plt.plot(SST[:, 0], SST[:, 3]/2509, 'r-', lw=1, label='SST') #, markersize=2)
plt.plot(SST15[:, 0], SST15[:, 3]/2509, 'b-', lw=1, label='SST15') #, markersize=2)


ax.set_xlabel(f'$x/δ$')
ax.set_ylabel(f'$P/P∞$')
plt.xlim(-0.06,0.06)
plt.legend()
plt.tight_layout()
plt.savefig('c_ramp_p.png')
plt.show()




