import numpy as np
import matplotlib.pyplot as plt

# Load all data files
exp_44 = np.loadtxt('0.44exp')
sst_44 = np.loadtxt('0.44sst')
nn_44 = np.loadtxt('0.44nn')
#prt_44 = np.loadtxt('0.44prt')
bp_44 = np.loadtxt('0.44bp')

exp_65 = np.loadtxt('0.65exp')
sst_65 = np.loadtxt('0.65sst')
nn_65 = np.loadtxt('0.65nn')
prt_65 = np.loadtxt('0.65prt')
bp_65 = np.loadtxt('0.65bp')

exp_80 = np.loadtxt('0.8exp')
sst_80 = np.loadtxt('0.8sst')
nn_80 = np.loadtxt('0.8nn')
prt_80 = np.loadtxt('0.8prt')
bp_80 = np.loadtxt('0.8bp')


exp_90 = np.loadtxt('0.9exp')
sst_90 = np.loadtxt('0.9sst')
nn_90 = np.loadtxt('0.9nn')
prt_90 = np.loadtxt('0.9prt')
bp_90 = np.loadtxt('0.9bp')

exp_96 = np.loadtxt('0.96exp')
sst_96 = np.loadtxt('0.96sst')
nn_96 = np.loadtxt('0.96nn')
prt_96 = np.loadtxt('0.96prt')
bp_96 = np.loadtxt('0.96bp')

exp_99 = np.loadtxt('0.99exp')
sst_99 = np.loadtxt('0.99sst')
nn_99 = np.loadtxt('0.99nn')
prt_99 = np.loadtxt('0.99prt')
bp_99 = np.loadtxt('0.99bp')



rc = {
    "font.family": 'Times New Roman',
    "mathtext.fontset": "stix",
    "font.size": 15 
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
legend_font = {
  
    'family': 'Times New Roman',  
    'size': 12,
    'weight':'bold' 
}

plt.rcParams.update(rc)

plt.rcParams["font.serif"] = ["Times New Roman"]+ plt.rcParams["font.serif"]

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

# Flatten the 2x3 axes array for easier indexing
axs = axs.flatten()

# Dataset labels
datasets = [(exp_44, sst_44, bp_44,'0.44'), 
            (exp_65, sst_65,bp_65,'0.65'), 
            (exp_80, sst_80, bp_80,'0.8'),
            (exp_90, sst_90, bp_90,'0.9'),
            (exp_96, sst_96, bp_96, '0.96'),
            (exp_99, sst_99,bp_99, '0.99')]

# Loop through each dataset and subplot
for i, (exp, sst,bp,label) in enumerate(datasets):
    ax = axs[i]  # Get the current subplot axis

    ax.plot(exp[:,2], -exp[:,4], color='black', marker='s', markersize=6, linestyle='', markerfacecolor='none', markeredgewidth=1.5,label=f'$Exp$')

    ax.plot((sst[:,0] - np.min(sst[:,0])) / (np.max(sst[:,0]) - np.min(sst[:,0])), -(sst[:,3] - 80510.081) / (0.5 * 0.935 * 291 * 291),
            '#d62728', lw=1.5, label=f'$SST$', linestyle='-')
    ax.plot((bp[:,0] - np.min(bp[:,0])) / (np.max(bp[:,0]) - np.min(bp[:,0])), -(bp[:,3] - 80510.081) / (0.5 * 0.935 * 291 * 291), color='blue', linestyle='-.', linewidth=1.5, label=f'$Improved\,SST$')  
        
    ax.set_title(f'$η={label}$', fontsize=12, x=0.45,y=0.83, pad=20, fontweight='bold', color='black',
                 backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.set_ylim(-0.8,1.5)  # Customize as needed
    ax.set_xlabel('$x/c$')  # Customize as needed
    ax.set_ylabel('$-C_{p}$')  # Customize as needed
    ax.legend(loc='upper right', ncol=1, prop=legend_font)
    ax.minorticks_on()
    #ax.grid(True)
plt.subplots_adjust(wspace=0.1, hspace=0.6)

# Adjust layout to avoid overlap
plt.tight_layout()
fig.savefig('cp.tiff',dpi=600 ,bbox_inches='tight')
# Show the plot
plt.show()



