import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = "Times New Roman"

plt.figure()
data_tree_1 = [7.86, None, None]
data_tree_2 = [0.52, 2.44, 2.01]
data_rand =  [0.28, 0.004, 0]
data_peace = [3.3, 0, 0]
data_rule = [8, 8.34, 3.40]
cat = ['coins', 'single', 'multi']
plt.ylabel('average score')

plt.scatter(cat, data_tree_1, label='tree_100k', marker='x', color='red', s=100)
plt.scatter(cat, data_tree_2, label='tree_1M', marker='+', color='red', s=100)
plt.scatter(cat, data_rand, label='random_agent', marker='1', color='black', s=100)
plt.scatter(cat, data_peace, label='peaceful_agent', marker='2', color='black', s=100)
plt.scatter(cat, data_rule, label='rule_based_agent', marker='*', color='black', s=100)
plt.legend()
plt.savefig('result.pdf')
plt.show()