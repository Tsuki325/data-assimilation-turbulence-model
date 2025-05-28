import numpy as np
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
def format_func(value, tick_number):
    return f'{value:.1f}'
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 300,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'font.size': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex': True,
    'figure.figsize': [5, 4],
    'font.family': 'serif',
}
#plt.rcParams.update(params)

# 加载实验数据
exp = np.loadtxt('34_hf.txt')
SST = np.loadtxt('wallHeatFlux_walls_constant1.raw', skiprows=2)
legend_font = {
    'family': 'Times New Roman',
    'size': 10,
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
    'size': 14,
}
rc = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 12
}

exp = np.loadtxt('cpexp.txt')
SST = np.loadtxt('p_walls_constant1.raw',skiprows=2)

output_dir = "iter results_CP"
os.makedirs(output_dir, exist_ok=True)
for ITER in (100,500,2000,3900):
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax = plt.subplots(figsize=(3, 2.2)) 
    plt.plot(exp[:, 0], exp[:, 1], label=r'$Exp$', color='black', marker='s', markersize=5, linestyle='', markerfacecolor='none', markeredgewidth=.8)
    base_color = (0.53, 0.81, 0.98, 1) 
    for i in range(0, 14):
        file_path = f'results_ensemble/sample_{i}/case1/postProcessing/sampleDict1/{ITER}/p_walls_constant.raw'
    #v/home/chen/compression_
        if os.path.exists(file_path):
            SSTNN = np.loadtxt(file_path, skiprows=2)
            # 只在第一个样本添加标签
            label = '$Ensemble$' if i == 1 else None
            plt.plot(SSTNN[:, 0]/0.027, SSTNN[:, 3]/23526, 
                     color=base_color, 
                     lw=1.5,
                     label=label)
    plt.plot(SST[:, 0]/0.027, SST[:, 3]/ 23526, 'r-', lw=1, label='$SST$') #, markersize=2)
    plt.xlim(-12, 6)
    plt.xticks(np.arange(-12, 6.1, 4))
    plt.yticks(np.arange(0, 5, 2))
    plt.ylim(0.8, 4.5)
    plt.text(0.06, 0.3, f'Case 1: step {int(ITER/100)}', fontsize=10, style='italic',family='Times New Roman', ha='left', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='grey', alpha=0.2))
    ax.minorticks_on()
    ax.set_xlabel(r'$x/δ$')
    ax.set_ylabel(r'$P_w/P_{ref}$', labelpad=1, weight='bold')
    plt.legend(loc='upper left', ncol=1, prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2Ma_CP_iter{ITER}.png'), dpi=900, bbox_inches='tight')





