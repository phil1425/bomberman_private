import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = "Times New Roman"



#filepath = 'agent_code/minimal_lstm_1/diagnostics.csv'
filepath1 = 'agent_code/minimal_tree_1/diagnostics_b.csv'
data1 = np.loadtxt(filepath1)
filepath2 = 'agent_code/minimal_tree_1/diagnostics_b2.csv'
data2 = np.loadtxt(filepath2)
filepath3 = 'agent_code/minimal_tree_1/diagnostics_b4.csv'
data3 = np.loadtxt(filepath3)

data = np.concatenate([data1, data2, data3], axis=0)


#filepath = 'agent_code/minimal_forest_1/diagnostics.csv'
#filepath = 'agent_code/minimal_linear_1/diagnostics.csv'
#filepath = 'agent_code/minimal_tree_1_benchmark_1/diagnostics.csv'

labels = ['steps', 'inv_moves', 'waited', 'crates', 'coins', 'kills', 'deaths', 'suicides']
factors = [1, 1, 1, 1, 1, 1, 1, 1]
'''
plt.figure()
for i in range(8):
    d = data[:,i]/factors[i]
    d = gaussian_filter1d(d, 10)
    plt.plot(d, label=labels[i])
    plt.legend()

plt.figure()
for i in range(8):
    d = data[:,i]/factors[i]
    d = d/data[:,0]
    d = gaussian_filter1d(d, 10)
    plt.plot(d, label=labels[i])
    plt.legend()

plt.show()
'''

plt.figure()
plt.subplot(221)
plt.plot(gaussian_filter1d(data[:,0], 1), c='grey', linewidth='1')
plt.plot(gaussian_filter1d(data[:,0], 40), label='steps per episode', c='black', linewidth='1')
plt.legend()
plt.subplot(222)
#plt.plot(gaussian_filter1d(data[:,6], 40), label='deaths per episode', c='red')
#plt.plot(gaussian_filter1d(data[:,7], 2), c='grey')
plt.plot(gaussian_filter1d(data[:,7], 40), label='suicides per episode', c='black')
plt.plot(gaussian_filter1d(data[:,5], 40), label='kills per episode', c='red')
plt.legend()
plt.subplot(223)
plt.xlabel('episodes')
plt.plot(gaussian_filter1d(data[:,1]/data[:,0], 1), c='grey')
plt.plot(gaussian_filter1d(data[:,2]/data[:,0], 1), c='lightcoral')
plt.plot(gaussian_filter1d(data[:,1]/data[:,0], 40), label='invalid move prob', c='black')
plt.plot(gaussian_filter1d(data[:,2]/data[:,0], 40), label='waited move prob', c='red')
plt.legend()
plt.subplot(224)
plt.xlabel('episodes')
plt.plot(gaussian_filter1d(data[:,4], 1), c='grey')
plt.plot(gaussian_filter1d(data[:,3]/20, 1), c='lightcoral')
plt.plot(gaussian_filter1d(data[:,4], 40), label='coins per episode', c='black')
plt.plot(gaussian_filter1d(data[:,3]/20, 40), label='crates per episode/20', c='red')
plt.legend(loc='best')
plt.savefig('multi.pdf')
plt.show()