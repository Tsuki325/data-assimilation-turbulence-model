import shap
import neuralnet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dafi import random_field as rf
from dafi.random_field import foam
from sklearn.decomposition import PCA

print(tf.__version__)


nInputs = 10
nOutputs = 1
nhlayers = 4
nnodes = 8
alpha = 0
variable_name = 'NN2input'
nn = neuralnet.NN(nInputs, nOutputs,nhlayers, nnodes, alpha)
w = np.loadtxt(f'{variable_name}/xa_119').mean(axis=1)
for i in range(8):
    w[i+80]=0
    w[i+152]=0
    w[i+224]=0
    w[i+296]=0
w[312]=0
w_shapes = neuralnet.weights_shape(nn.trainable_variables)
w_reshape = neuralnet.reshape_weights(w, w_shapes)
nn.set_weights(w_reshape)


lamda1 = rf.foam.read_scalar_field(f'{variable_name}/lambda1')
lamda2 = rf.foam.read_scalar_field(f'{variable_name}/lambda2')
Fs = rf.foam.read_scalar_field(f'{variable_name}/Fs')
nk = rf.foam.read_scalar_field(f'{variable_name}/nk')
Mg = rf.foam.read_scalar_field(f'{variable_name}/Mg')
TuM = rf.foam.read_scalar_field(f'{variable_name}/TuM')
q2 = rf.foam.read_scalar_field(f'{variable_name}/q2')
q3 = rf.foam.read_scalar_field(f'{variable_name}/q3')
I1 = rf.foam.read_scalar_field(f'{variable_name}/I2_1')
I2 = rf.foam.read_scalar_field(f'{variable_name}/I2_3')

inputScalarNumber=10           
input_scalars = np.empty((len(lamda1),inputScalarNumber))


Ma=2.84
input_scalars[:,0] = lamda1
input_scalars[:,1] = -lamda2
input_scalars[:,2] = Fs
input_scalars[:,3] = np.abs(nk)
input_scalars[:,4] = TuM/Ma
input_scalars[:,5] = Mg/Ma
input_scalars[:,6] = q2
input_scalars[:,7] = -q3
input_scalars[:,8] = I1
input_scalars[:,9] = I2
th_min = np.loadtxt(f'{variable_name}/input_preproc_stat_0_11900')
th_max = np.loadtxt(f'{variable_name}/input_preproc_stat_1_11900')
input_scalars = (input_scalars-th_min)/(th_max-th_min)
print(len(input_scalars))

def rel_dis(a,b):
    dis = np.sum(np.abs(a - b) / np.maximum(np.abs(a), np.abs(b)))
    return dis

#sparse input
input_sp = input_scalars[0,:].reshape(1, inputScalarNumber)
delta = 8
for i in range(len(input_scalars)):
    j = input_sp.shape[0]
    print(i,j)
    if all( rel_dis(input_scalars[i,:],input_sp[n,:]) > delta for n in range(j)):
        input_sp = np.concatenate((input_sp,input_scalars[i,:].reshape(1, inputScalarNumber)),axis=0)
        
np.savetxt('input_sp',input_sp)        
        
   
x_df1 = input_sp


explainer = shap.KernelExplainer(nn.predict, x_df1)
shap_values = explainer.shap_values(x_df1)
shap_values_explanation = shap.Explanation(shap_values, base_values=explainer.expected_value, data=x_df1)
#shap_values_explanation.base_values = np.full((inputScalarNumber), shap_values_explanation.base_values[0],dtype=np.float32)
shap_values_explanation.values = np.array(shap_values_explanation.values[0],dtype=np.float32)
shap_values_explanation.data = np.array(shap_values_explanation.data,dtype=np.float32)
print(shap_values_explanation.data,shap_values_explanation.values)

features = [f"Feature {i}" for i in range(10)]
shap_values = shap_values_explanation.values.T#np.random.randn(10, 100)
top_rows = shap_values[:2]
shap_values = np.vstack([shap_values[2:], top_rows])

feature_values = shap_values_explanation.data.T#np.random.randn(10, 100)
top_rows = feature_values[:2]
feature_values = np.vstack([feature_values[2:], top_rows])

# 计算每个特征SHAP值绝对值的平均值
shap_abs_mean = np.mean(np.abs(shap_values), axis=1)

# 设置字体为Times New Roman，并设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20  # 全局字体大小
plt.rcParams["axes.titlesize"] = 20  # 标题字体大小
plt.rcParams["axes.labelsize"] = 20 # 轴标签字体大小
plt.rcParams["xtick.labelsize"] = 20  # x轴刻度标签字体大小
plt.rcParams["ytick.labelsize"] = 20  # y轴刻度标签字体大小
plt.rcParams["legend.fontsize"] = 20  # 图例字体大小
plt.rcParams["figure.titlesize"] = 20  # 图形标题字体大小

# 创建图形和两个子图
fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
fig.set_size_inches(12,12) 
# 设置colormap
cmap = plt.get_cmap("BuRd")

for i in range(len(features)):
    # 计算颜色
    colors = cmap((feature_values[i] - feature_values[i].min()) / (feature_values[i].max() - feature_values[i].min()))
    
    # 绘制散点图
    ax_scatter.scatter(shap_values[i], np.full(shap_values[i].shape, i), color=colors, s=10, alpha=0.6, edgecolors='w', linewidth=0.5)

# 设置散点图的x轴范围
ax_scatter.set_xlim(-np.max(np.abs(shap_values)), np.max(np.abs(shap_values)))

# 设置y轴
ax_scatter.set_yticks(np.arange(len(features)))
ax_scatter.set_yticklabels(features, fontsize=20)

# 绘制柱状图
bar_width = 0.7
bar_y = np.arange(len(features))
ax_bar.barh(bar_y, shap_abs_mean, color=cmap(0.8), height=bar_width, align='center', alpha=1)

# 隐藏柱状图的y轴标签和刻度
ax_bar.set_yticks([])
ax_bar.set_yticklabels([])
ax_bar.set_yticklabels([])
ax_bar.set_xlabel("Mean of the absolute SHAP values", fontsize=20)

# 设置标题和标签
ax_scatter.set_xlabel("SHAP value", fontsize=20)


# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=plt.cm.BuRd, norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm, ax=ax_scatter)
cbar.set_label("Feature value", fontsize=16)
cbar.ax.yaxis.set_label_position('right')  

#plt.tight_layout()

plt.show()
fig.savefig('NN2.png',format='png',dpi=600)


