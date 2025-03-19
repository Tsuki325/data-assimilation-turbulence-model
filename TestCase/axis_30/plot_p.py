from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pyplot as plt
import pdb
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
    'style': 'italic',
    'size': 14,
    #'weight':'bold' 
} 
rc = {
    "font.family": 'Times New Roman',
    "mathtext.fontset": "stix",
    "font.size": 18 
}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"]+ plt.rcParams["font.serif"]
fig, ax = plt.subplots()
fig.set_size_inches(6, 5) 
 

exp = np.loadtxt('30_p.txt')
SST = np.loadtxt('p_walls_constant1.raw',skiprows=2)
SSTBP = np.loadtxt('p_walls_constant.raw',skiprows=2)
a=100
b=140
#x to S  1.15  30度  romp
plt.plot(exp[:,0], exp[:,1],label=r'$Exp$' ,color='black',marker='s',markersize=6,linestyle='',markerfacecolor='none',markeredgewidth=1.5 )
plt.plot(SST[:, 0]*1.13*100, SST[:, 3]/576, '#d62728', lw=1.5, label='$SST$') #, markersize=2)
plt.plot(SSTBP[:, 0]*1.15*100, SSTBP[:, 3]/576, color='blue',linestyle='-.', linewidth=1.5,label='$Improved\,SST$') #, markersize=2)
ax.set_xlabel(f'$s(cm)$',labelpad=-4,weight='bold')
ax.set_ylabel(r'$P_w/P_{ref}$',labelpad=1,weight='bold')
ax.minorticks_on() 
plt.xlim(-10,15)
plt.ylim(0,35)
plt.legend(loc='upper left', ncol=1, prop=legend_font)
ax_inset = inset_axes(ax, width="40%", height="30%", bbox_to_anchor=(-0.4, -0.3, 0.8, 0.8), bbox_transform=ax.transAxes) # Adjust the size and location as needed

# Plot data on the inset
ax_inset.plot(exp[:, 0], exp[:, 1],label=r'Exp' ,color='black',marker='s',markersize=6,linestyle='',markerfacecolor='none',markeredgewidth=1.5 )
ax_inset.plot(SST[:, 0]*1.15*100, SST[:, 3]/576, '#d62728', lw=1.5, label='SST') #, markersize=2)
ax_inset.plot(SSTBP[:,0]*1.15*100, SSTBP[:, 3]/576, color='blue', linestyle='-.', linewidth=1.5,label='$SST+M_{3}$')

x_range = (-10, 0)  # Example x-axis range
y_range = (0, 3)  # Example y-axis range

ax_inset.set_xlim(x_range)
ax_inset.set_ylim(y_range)
ax_inset.set_xticklabels([])
ax_inset.set_yticklabels([])
ax_inset.tick_params(left=False, right=False, top=False, bottom=False)  # Hide inset ticks

#ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

# Draw rectangle in the main plot to indicate the zoomed region
rect = plt.Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                     edgecolor='black', facecolor='none', linestyle='--')
ax.add_patch(rect)

# Connect the rectangle to the inset
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig('c_rampaxis_p.png',dpi=600)
plt.show()





