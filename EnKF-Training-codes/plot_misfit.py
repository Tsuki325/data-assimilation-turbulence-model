from matplotlib import font_manager
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt

# 原始日志文本
log_text = """
  Iteration: 0
      TensorFlow ... 2.50s
      TensorFlow ... 5.25s
      Ensemble of forecast ... 1095.77s
      Data assimilation analysis ... 0.01s
      misfit = ... 3.1569234997568283

  Iteration: 1
      TensorFlow ... 2.29s
      TensorFlow ... 5.03s
      Ensemble of forecast ... 1079.06s
      Data assimilation analysis ... 0.01s
      misfit = ... 2.8128256927589455

  Iteration: 2
      TensorFlow ... 2.34s
      TensorFlow ... 5.14s
      Ensemble of forecast ... 1068.60s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.9532820297678661

  Iteration: 3
      TensorFlow ... 2.26s
      TensorFlow ... 5.01s
      Ensemble of forecast ... 1081.60s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.808343674753906

  Iteration: 4
      TensorFlow ... 2.26s
      TensorFlow ... 5.03s
      Ensemble of forecast ... 1076.72s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.8202678995138235

  Iteration: 5
      TensorFlow ... 2.28s
      TensorFlow ... 5.05s
      Ensemble of forecast ... 1086.97s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.769243159282256

  Iteration: 6
      TensorFlow ... 2.28s
      TensorFlow ... 4.99s
      Ensemble of forecast ... 1081.10s
      Data assimilation analysis ... 0.00s
      misfit = ... 1.8207181623991717

  Iteration: 7
      TensorFlow ... 2.33s
      TensorFlow ... 5.20s
      Ensemble of forecast ... 1082.94s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.8090595885843916

  Iteration: 8
      TensorFlow ... 2.24s
      TensorFlow ... 4.96s
      Ensemble of forecast ... 1077.74s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.49631893518224

  Iteration: 9
      TensorFlow ... 2.24s
      TensorFlow ... 5.45s
      Ensemble of forecast ... 1070.15s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.279754392689781

  Iteration: 10
      TensorFlow ... 2.29s
      TensorFlow ... 5.29s
      Ensemble of forecast ... 1087.37s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.2781963187674072

  Iteration: 11
      TensorFlow ... 2.41s
      TensorFlow ... 6.97s
       Ensemble of forecast ... 1086.69s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.3579410344796645

  Iteration: 12
      TensorFlow ... 2.29s
      TensorFlow ... 6.79s
      Ensemble of forecast ... 1101.25s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.4489999255020432

  Iteration: 13
      TensorFlow ... 2.22s
      TensorFlow ... 5.13s
      Ensemble of forecast ... 1108.48s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.5410735862378773

  Iteration: 14
      TensorFlow ... 2.19s
      TensorFlow ... 4.84s
      Ensemble of forecast ... 1102.15s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.4904530630242292

  Iteration: 15
      TensorFlow ... 2.29s
      TensorFlow ... 4.96s
      Ensemble of forecast ... 1103.81s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.4505376664463845

  Iteration: 16
      TensorFlow ... 2.28s
      TensorFlow ... 6.17s
      Ensemble of forecast ... 1093.24s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.3714151830608563

  Iteration: 17
      TensorFlow ... 2.32s
      TensorFlow ... 6.90s
      Ensemble of forecast ... 1092.36s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.3158251842991067

  Iteration: 18
      TensorFlow ... 2.39s
      TensorFlow ... 5.11s
      Ensemble of forecast ... 1095.78s
      Data assimilation analysis ... 0.02s
      misfit = ... 1.2010124093489842

  Iteration: 19
      TensorFlow ... 2.36s
      TensorFlow ... 5.74s
      Ensemble of forecast ... 1099.60s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1736662873335226

  Iteration: 20
      TensorFlow ... 2.45s
      TensorFlow ... 5.24s
      Ensemble of forecast ... 1105.71s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.167274489059852

  Iteration: 21
      TensorFlow ... 2.34s
      TensorFlow ... 5.16s
      Ensemble of forecast ... 1090.56s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.0851375314783434

  Iteration: 22
      TensorFlow ... 2.36s
      TensorFlow ... 5.17s
      Ensemble of forecast ... 1097.48s
      Data assimilation analysis ... 0.02s
      misfit = ... 1.0694523480101257

  Iteration: 23
      TensorFlow ... 2.31s
      TensorFlow ... 5.07s
      Ensemble of forecast ... 1093.88s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1459470595811858

  Iteration: 24
      TensorFlow ... 2.36s
      TensorFlow ... 5.10s
      Ensemble of forecast ... 1098.91s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.230137275778403

  Iteration: 25
      TensorFlow ... 2.24s
      TensorFlow ... 5.56s
      Ensemble of forecast ... 1088.53s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1892517335417907

  Iteration: 26
      TensorFlow ... 2.33s
      TensorFlow ... 6.91s
      Ensemble of forecast ... 1102.47s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.2019801777469241

  Iteration: 27
      TensorFlow ... 2.31s
      TensorFlow ... 6.93s
      Ensemble of forecast ... 1090.43s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.2977716196471727

  Iteration: 28
      TensorFlow ... 2.30s
      TensorFlow ... 6.92s
      Ensemble of forecast ... 1096.51s
      Data assimilation analysis ... 0.02s
      misfit = ... 1.245131999302515

  Iteration: 29
      TensorFlow ... 2.47s
      TensorFlow ... 7.26s
      Ensemble of forecast ... 1091.80s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.2083945093945025

  Iteration: 30
      TensorFlow ... 2.31s
      TensorFlow ... 6.86s
      Ensemble of forecast ... 1084.98s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.160214599259462

  Iteration: 31
      TensorFlow ... 2.28s
      TensorFlow ... 6.88s
      Ensemble of forecast ... 1094.40s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1974233132432655

  Iteration: 32
      TensorFlow ... 3.01s
      TensorFlow ... 8.15s
      Ensemble of forecast ... 1097.51s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.210504653266183

  Iteration: 33
      TensorFlow ... 3.87s
      TensorFlow ... 8.95s
      Ensemble of forecast ... 1097.49s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1089021762108013

  Iteration: 34
      TensorFlow ... 3.34s
      TensorFlow ... 7.95s
      Ensemble of forecast ... 1102.89s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1893576363991498

  Iteration: 35
      TensorFlow ... 3.32s
      TensorFlow ... 7.95s
      Ensemble of forecast ... 1101.67s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1615324345538753

  Iteration: 36
      TensorFlow ... 3.36s
      TensorFlow ... 7.94s
      Ensemble of forecast ... 1102.82s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.1668709662812722

  Iteration: 37
      TensorFlow ... 2.32s
      TensorFlow ... 5.17s
      Ensemble of forecast ... 1093.81s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.192003556611745

  Iteration: 38
      TensorFlow ... 2.31s
      TensorFlow ... 5.66s
      Ensemble of forecast ... 1093.63s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.223038928414941

  Iteration: 39
      TensorFlow ... 2.50s
      TensorFlow ... 7.28s
      Ensemble of forecast ... 1099.37s
      Data assimilation analysis ... 0.01s
      misfit = ... 1.123347166626602


"""

# 使用正则表达式提取所有misfit值
misfits = re.findall(r'misfit = \.\.\. (\d+\.\d+)', log_text)
misfits = [float(m) for m in misfits]  # 转换为浮点数

misfits = list(map(lambda x: x ** 2, misfits))
# 生成迭代次数列表
iterations = list(range(len(misfits)))
# Load the custom font (Edwardian Script ITC)
font_path = 'EdwardianScriptITC.ttf'
prop = font_manager.FontProperties(fname=font_path)

# Font properties for the labels and legend
font2 = {'family': 'Times New Roman',
         'style': 'italic',
         'weight': 'bold',
         'size': 38,  # Increased font size
         }

font2 = {'family': 'Times New Roman',
         'style': 'italic',
         'size': 30,  # Increased font size
         }
         
font3 = {'family': 'Times New Roman',
         'style': 'italic',
         'weight': 'bold',
         'size': 38,  # Increased font size
         }
legend_font = {'family': 'Times New Roman',
               'style': 'italic',
               'weight': 'bold',
               'size': 20,  # Increased font size
               }

# Create figure and axis
fig, ax1 = plt.subplots()
fig.set_size_inches(6, 4)  # Aspect ratio of the plot
# ... (前面的导入和日志文本保持不变)

# 找到最低点和初始值
min_misfit = min(misfits)
min_index = misfits.index(min_misfit)
initial_misfit = misfits[0]

# 计算下降百分比
reduction_percent = (initial_misfit - min_misfit) / initial_misfit * 100

# ... (前面的绘图设置保持不变)

# 在最低点添加标注
ax1.annotate(f'{reduction_percent:.1f}% reduction',
             xy=(min_index, min_misfit),
             xytext=(min_index-10, min_misfit+3),  # 调整文本位置
             arrowprops=dict(arrowstyle='->', 
                           connectionstyle='arc3',
                           color='grey',
                           linewidth=2),
             fontsize=20,
             fontname='Times New Roman',
             bbox=dict(boxstyle='round', 
                     facecolor='white', 
                     edgecolor='gray', 
                     alpha=0.9))

# ... (后面的图例和保存代码保持不变)
# Load data



# Plot the data
ax1.plot(iterations, misfits, 
         linestyle='-.',          # 设置虚线样式
         color='blue',            # 线条颜色
         linewidth=3,             # 线宽
         marker='x',             # 标记类型（x）
         markevery=1,            # 每个数据点都显示标记
         markersize=6,            # 标记大小
         markeredgecolor='red',   # 标记边缘颜色
         markeredgewidth=2,     # 标记边缘宽度
         markerfacecolor='none',  # 标记填充色（无）
         label='Inverse convergence history')  # 图例标签

# Set tick parameters
ax1.set_xticks(np.arange(0, 41, 10))
ax1.set_yticks(np.arange(0, 10.1, 2))  # Increased tick font size
ax1.set_xlim(0, 40)
#ax1.set_ylim(0, 1)
# Set x and y ticks


# Set font properties (family) for x and y axis tick labels
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')  # Set font family to Times New Roman

for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')  # Set font family to Times New Roman

# Set labels
ax1.set_xlabel(r"Iteration step", font2, labelpad=2)  # Increased label padding
ax1.set_ylabel(r'J', font2,labelpad=0)  # Increased font size

# Set spine widths and colors for axis lines
ax1.spines['top'].set_linewidth(2)
ax1.spines['right'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)

# Set color for spines (axis lines)
ax1.spines['top'].set_color('black')
ax1.spines['right'].set_color('black')
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')

# Set color for ticks (axis labels)
ax1.tick_params(axis='x', labelsize=30, colors='black')  # Set x-axis tick color
ax1.tick_params(axis='y', labelsize=30, colors='black')  # Set y-axis tick color
plt.grid(True, linestyle='-', linewidth=1.2, color='black', alpha=0.2)
plt.legend(loc='upper right', ncol=1, prop=legend_font)
plt.tight_layout()

# Show plot
plt.show()

# Save plot as TIFF
fig.savefig('misfit.png', format='png', dpi=600, bbox_inches='tight')

