import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

filepath = 'agent_code/minimal_tree_1/diagnostics.csv'
data = np.loadtxt(filepath)

labels = ['steps', 'inv_moves', 'waited', 'crates', 'coins', 'kills', 'deaths', 'suicides']
factors = [1, 1, 1, 1, 1, 1, 1, 1]

plt.figure()
for i in range(1, 8):
    d = data[:,i]/factors[i]
    d = gaussian_filter1d(d, 10)
    plt.plot(d, label=labels[i])
    plt.legend()

plt.figure()
for i in range(1, 8):
    d = data[:,i]/factors[i]
    d = d/data[:,0]
    d = gaussian_filter1d(d, 10)
    plt.plot(d, label=labels[i])
    plt.legend()

plt.show()