
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


exp = np.loadtxt('cpexp.txt')
SST = np.loadtxt('p_walls_constant.raw',skiprows=2)

fig, ax = plt.subplots()

#plt.plot(exp[:, 0], exp[:, 1], 'ko', markersize=7, fillstyle='none', label='exp')
plt.plot(SST[:, 0], SST[:, 3]/2509, 'r-', lw=1, label='SST') #, markersize=2)
print(SST[:, 3])
'''
plt.plot(SST[45:180,0]/0.02, SST[45:180, 3]/ 23526, 'g-', lw=3, label='SST_inter') #, markersize=2)
interp_func = interpolate.interp1d(exp[:, 0], exp[:, 1])

xc = SST[45:180,0]/0.02

exp_interplot = interp_func(xc)
EXPToMesh=np.vstack([xc,exp_interplot]).T
print(SST[45:180,0]/0.02)
plt.plot(xc, exp_interplot ,  'b-', lw=3, label='expToMesh')
np.savetxt('EXPToMesh',EXPToMesh)
'''
ax.set_xlabel(f'$x/δ$')
ax.set_ylabel(f'$P/P∞$')
plt.xlim(-0.05,0.1)
plt.legend()
plt.tight_layout()
plt.savefig('c_ramp_p.png')
plt.show()




