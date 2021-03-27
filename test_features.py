from features import FeatureSelector
import pickle
import matplotlib.pyplot as plt

states = pickle.load(open('featureset.pt', 'rb'))

for state in states:
    selector = FeatureSelector(state)
    selector.sum_of_crates_around_agent()
    print(selector.features)

    plt.figure()
    vis_field = state['field'].copy()
    x, y = state['self'][3]
    vis_field[x, y] = 4
    #for bomb in state['bombs']:
    #    bx, by = bomb[0]
    #    vis_field[bx, by] = 2
    for bomb in state['coins']:
        bx, by = bomb
        vis_field[bx, by] = 2

    for agent in state['others']:
        bx, by = agent[3]
        vis_field[bx, by] = 3

    plt.imshow(vis_field)
    plt.show()