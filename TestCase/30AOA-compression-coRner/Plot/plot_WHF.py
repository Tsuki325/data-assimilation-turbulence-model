
import numpy as np
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pyplot as plt
import pdb
# set plot properties
legend_font = {
  
    'family': 'Times New Roman',  
    'style': 'italic',
    'size': 14,
    #'weight':'bold' 
}    
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

rc = {
    "font.family": "serif", 
    "mathtext.fontset": "stix",
    #"font.weight": "bold",  
    "font.size": 18 
}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"]+ plt.rcParams["font.serif"]
fig, ax = plt.subplots()
fig.set_size_inches(6, 5) 
exp = np.loadtxt('30_hf.txt')
SST = np.loadtxt('wallHeatFlux_walls_constant.raw',skiprows=2)
SSTNN = np.loadtxt('wallHeatFlux_walls_constant1.raw',skiprows=2)
a=100
b=220
plt.plot(exp[:, 0]/1.2/0.0072, exp[:, 1],label=r'Exp' ,color='black',marker='s',markersize=7,linestyle='',markerfacecolor='none',markeredgewidth=1.5) 

from scipy.interpolate import UnivariateSpline


x = SST[:,0]/0.0072
y= -SST[:,3]/64000
mask_positive = x > 0
x_positive = x[mask_positive]
y_positive = y[mask_positive]

spline = UnivariateSpline(x_positive, y_positive, s=8) 
smoothed_positive_data = spline(x_positive)

mask_negative = x < 0
x_negative = x[mask_negative]
y_negative = y[mask_negative]

x_combined = np.concatenate((x_negative, x_positive))
y_combined = np.concatenate((y_negative, smoothed_positive_data))
plt.plot(x_combined,y_combined, 'r-', lw=1.5, label='SST') #, markersize=2)

################################

x = SSTNN[:,0]/0.0072
y = -SSTNN[:,3]/64000
mask_positive = x > 0
x_positive = x[mask_positive]
y_positive = y[mask_positive]
spline = UnivariateSpline(x_positive, y_positive, s=8) 
smoothed_positive_data = spline(x_positive)

mask_negative = x < 0
x_negative = x[mask_negative]
y_negative = y[mask_negative]

x_combined1 = np.concatenate((x_negative, x_positive))
y_combined1 = np.concatenate((y_negative, smoothed_positive_data))
plt.plot(x_combined1,y_combined1, color='blue',linestyle='-.', linewidth=2,label=r'$Improved \,SST$') #, markersize=2)
###########################################



ax.minorticks_on() 
ax.set_xlabel(f'$x/δ$')
ax.set_ylabel(r'$Q_w/Q_{ref}$',labelpad=1,weight='bold')
plt.xlim(-8,10)
plt.ylim(0,40)
plt.legend(loc='upper left', ncol=1, prop=legend_font)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
ax_inset = inset_axes(ax, width="40%", height="30%", bbox_to_anchor=(-0.4, -0.3, 0.8, 0.8), bbox_transform=ax.transAxes) # Adjust the size and location as needed

ax_inset.plot(exp[:, 0]/0.0072, exp[:, 1],label=r'$Exp$' ,color='black',marker='s',markersize=7,linestyle='',markerfacecolor='none',markeredgewidth=1.5 )
ax_inset.plot(x_combined,y_combined, 'r', lw=2, label=f'$SST$') #, markersize=2)
ax_inset.plot(x_combined1,y_combined1, color='blue',linestyle='-.', linewidth=2,label=r'$Improved \,SST$') #, markersize=2)


# Set limits and labels for the inset
# Define your specific range for x and y axes for the inset
x_range = (-5, 0)  # Example x-axis range
y_range = (0, 4)  # Example y-axis range

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
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig('c_ramp_HF.tiff',dpi=600,bbox_inches='tight')
plt.show()




