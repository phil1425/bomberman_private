from sklearn import tree
import matplotlib.pyplot as plt
import pickle
from tree_model import TreeModel
from dtreeviz.trees import dtreeviz
import numpy as np

cols = ["Sum of crates radius 1", "Sum of crates radius 2", #"Agent in explosion zone",
        "Crates explode if bomb here", "Agents explode if bomb here", #"Straight bomb distance", "Straight bomb cooldown",
        "Distance next agent", "Next agent rel x", "Next agent rel y", "Crates around next agent", "Distance next coin", "Next coin rel x",
        "Next coin rel y", "Crates around next coin", "Bomb possible",
        "Sum all crates", "Agent pos rel to wall", "Free space down", "Free space right", "Free space up", "Free space left",
        "Survivable spaces down", "Survivable spaces right", "Survivable spaces up", "Survivable spaces left", "Survivable spaces wait"]

labels = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

model = TreeModel(1000, (21,), (6,), 1)
model.load('agent_code/minimal_tree_1/actor.p')
X = np.load('agent_code/minimal_tree_1/actor_X.npy')
y = np.load('agent_code/minimal_tree_1/actor_y.npy')[:,0]

plt.figure(figsize=(25, 20))
tree.plot_tree(model.model, feature_names=cols, class_names=y, max_depth=10, fontsize=4)
plt.show()
#print(tree.export_text(model.model)[:10000])