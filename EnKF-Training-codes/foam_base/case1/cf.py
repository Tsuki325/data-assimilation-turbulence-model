import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
legend_font = {
  
    'family': 'Times New Roman',  
    'style': 'italic',
    'size': 22,
    'weight':'bold' 
}    
font2 = {'family' : 'Times New Roman',
    'style': 'italic',
    'weight':'bold',
'size' : 25,
}
font = {'family' : 'Times New Roman',
    #'weight':'bold',
'size' : 18,
}
font3 = {'family' : 'Times New Roman',
    'weight':'bold',
'size' : 18,
}
fig, ax = plt.subplots()
fig.set_size_inches(12, 10.7) 
cfsst=np.genfromtxt('wallShearStress_walls_constant.raw',comments='#')
exp = np.loadtxt('cfexp.txt')
plt.plot(exp[:,0],exp[:,1],label=r'Exp', color='green',marker='^',markersize=7.5,linestyle='',markerfacecolor='green')
plt.plot(cfsst[:,0]/0.023,-cfsst[:,3]/0.82/570/570/0.5,label=r'k-' '\u03C9 SST', color='black',linestyle='-', linewidth=2)



plt.xlim(-8,8)
plt.tick_params(labelsize=20) 
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font)
ax.set_xlabel(r"x/δ",font2,labelpad=4)
ax.set_ylabel("C$_f$",font2,labelpad=4)
#ax.legend(loc="upper center",ncol=1,frameon=True,prop=legend_font,bbox_to_anchor=(0.28, 1)) 
ax.minorticks_on() 
plt.show()
fig.savefig("cf.png", format='png', dpi=600)
plt.show()

