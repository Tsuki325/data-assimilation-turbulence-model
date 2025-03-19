
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


exp = np.loadtxt('swtblis13_cp.txt')
SST = np.loadtxt('p_walls_constant1.raw',skiprows=2)
SSTNN = np.loadtxt('p_walls_constant.raw',skiprows=2)
fig, ax = plt.subplots()

plt.plot(exp[:, 0], exp[:, 1], 'ko', markersize=5, fillstyle='none', label='exp')
plt.plot(SST[:, 0]/0.01894, SST[:, 3]/21800, 'r-', lw=1, label='SST') #, markersize=2)
plt.plot(SSTNN[:, 0]/0.01894, SSTNN[:, 3]/21800, 'r-', lw=1, label='SSTNN') #, markersize=2)


ax.set_xlabel(f'$x/δ$')
ax.set_ylabel(f'$P/P∞$')
plt.xlim(-12,6)
#plt.ylim(0,60)
plt.legend()
plt.tight_layout()
plt.savefig('c_ramp_p.png')
plt.show()




