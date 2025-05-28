import numpy as np
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_smoothed_data(x, y, label, color, linestyle, linewidth, alpha=1.0):
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
    plt.plot(x_combined/0.0085, y_combined, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha)


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

output_dir = "iter results"
os.makedirs(output_dir, exist_ok=True)
for ITER in (100,500,2000,3900):
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax = plt.subplots(figsize=(3, 2.2)) 
    plt.plot(exp[:, 0] / 1.2 / 0.0085, exp[:, 1], label=r'$Exp$', color='black', marker='s', markersize=5, linestyle='', markerfacecolor='none', markeredgewidth=.8)
    
    base_color = (0.53, 0.81, 0.98, 0.3) # 带透明度的绿色 (RGBA)天蓝色：(0.53, 0.81, 0.98, 0.3)皇家蓝：(0.25, 0.41, 0.88, 0.3)深空蓝：(0.0, 0.75, 1.0, 0.3)
    for i in (0,1,2,3,4,5,6,7,8,9,11,12,13):#,6,7,8,9,10,11,12,13, 14):
        file_path = f'results_ensemble/sample_{i}/case2/postProcessing/sampleDict1/{ITER}/wallHeatFlux_walls_constant.raw'
        if os.path.exists(file_path):
            SSTNN = np.loadtxt(file_path, skiprows=2)
            label = '$Ensemble$' if i == 1 else None
            plot_smoothed_data(SSTNN[:, 0], -SSTNN[:, 3]/61000,label, base_color, '-', .9)
    plot_smoothed_data(SST[:, 0], -SST[:, 3]/61000, '$SST$','r',  '-'  ,.8)

    # 设置坐标轴和范围
    ax.minorticks_on()
    ax.set_xlabel(r'$x/δ$')
    ax.set_ylabel(r'$Q_w/Q_{ref}$', labelpad=1, weight='bold')
    plt.legend(loc='upper left', ncol=1, prop=legend_font)
    plt.xlim(-6, 6)
    plt.xticks(np.arange(-8, 7, 4))
    plt.ylim(0, 45)
    plt.yticks(np.arange(0, 45, 20))

    # 添加标题，显示当前的迭代步
    plt.text(0.07, 0.3, f'Case 2: step {int(ITER/100)}', fontsize=10, style= 'italic',family='Times New Roman', ha='left', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='grey', alpha=0.2))





    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'9Ma_HF_iter{ITER}.png'), dpi=900, bbox_inches='tight')

