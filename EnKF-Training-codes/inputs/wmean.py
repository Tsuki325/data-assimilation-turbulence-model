import numpy as np
w=np.loadtxt('xa_132').mean(axis=1)
np.savetxt('w.6148_learned',w)
