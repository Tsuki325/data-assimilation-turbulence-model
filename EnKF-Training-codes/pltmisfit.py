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

"""

# 使用正则表达式提取所有misfit值
misfits = re.findall(r'misfit = \.\.\. (\d+\.\d+)', log_text)
misfits = [float(m) for m in misfits]  # 转换为浮点数

misfits = list(map(lambda x: x ** 2, misfits))
# 生成迭代次数列表
iterations = list(range(len(misfits)))

# 绘制misfit变化曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, misfits, 'b-o', linewidth=2, markersize=8)
plt.title('Misfit Value over Iterations', fontsize=14)
plt.xlabel('Iteration Number', fontsize=12)
plt.ylabel('Misfit Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(iterations)   #显示所有迭代刻度

# 标记最小值
min_misfit = min(misfits)
min_index = misfits.index(min_misfit)
plt.scatter(min_index, min_misfit, color='red', s=100, zorder=5)
plt.annotate(f'Min: {min_misfit:.3f}', 
             xy=(min_index, min_misfit),
             xytext=(min_index+1, min_misfit+0.2),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.tight_layout()
plt.show()
