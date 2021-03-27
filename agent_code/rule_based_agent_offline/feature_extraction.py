import numpy as np
import pickle

PADCONST = 3

featurepickle = open("./featureset_large.pt", "rb")
labelpickle = open("./labelset_large.pt", "rb")

gamestates = pickle.load(featurepickle)
labels = pickle.load(labelpickle)

all_features = []
cols = ["Step Number", "Sum of crates radius 1", "Sum of crates radius 2", "Agent in explosion zone",
        "Crates explode if bomb here", "Agents explode if bomb here", "Straight bomb distance", "Straight bomb cooldown",
        "Distance next agent", "Next agent rel x", "Next agent rel y", "Coins in radius 3", "Distance next coin", "Next coin rel x",
        "Next coin rel y", "Bomb possible", "Agent score", "Other Agent 1 score", "Other Agent 2 score", "Other Agent 3 score",
        "Sum all crates", "Crates around next coin", "Crates around next agent", "Agent at left/right border", "Agent at top/bottom border"]

for idx, game_state in enumerate(gamestates):

    # if idx == 2:
    #     break
    features = []
    
    print(f"##########################\n\n Extracting feature {idx}\n\n##########################")


    field = game_state['field']
    walls = (game_state['field'] == -1).astype(int)
    walls_padded = np.pad(walls, PADCONST, 'edge')
    crates = (game_state['field'] == 1).astype(int)
    crates_padded = np.pad(crates, PADCONST, 'edge')

    # print(field, "\n", walls, "\n", walls_padded, "\n", crates, "\n", crates_padded)

    agent_pos = game_state['self'][3]
    agent_pos_x = agent_pos[0]
    agent_pos_y = agent_pos[1]
    agent_pos_x_pad = agent_pos_x + PADCONST
    agent_pos_y_pad = agent_pos_y + PADCONST


    # Current Step Number
    step_number = game_state['step']
    features.append(step_number)

    # Sum of crates in radius 1 around agent
    features.append(np.sum(crates_padded[agent_pos_x_pad-1:agent_pos_x_pad+1, agent_pos_y_pad-1:agent_pos_y_pad+1]))

    # Sum of crates in radius 2 around agent
    features.append(np.sum(crates_padded[agent_pos_x_pad-2:agent_pos_x_pad+2, agent_pos_y_pad-2:agent_pos_y_pad+2]))

    # Agent in explosion zone (bool)
    bomb_map = np.zeros((17+2*PADCONST, 17+2*PADCONST))
    bomb_cooldowns = [x[1] for x in game_state['bombs']]
    bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))

    for idx in bomb_idx_sorted:
        bomb = game_state['bombs'][idx]
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        if np.sum(walls[b_x-1:b_x+1, b_y]) == 2:
            bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = bomb[1]
        elif np.sum(walls[b_x, b_y-1:b_y+1]) == 2:
            bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = bomb[1]
        else:
            bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = bomb[1]
            bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = bomb[1]
    
    features.append(bomb_map[agent_pos_x_pad, agent_pos_y_pad])

    # Crates that explode if bomb is placed at current location
    custom_bomb_map = np.zeros((17+2*PADCONST, 17+2*PADCONST))

    b_x = agent_pos_x_pad
    b_y = agent_pos_y_pad
    if np.sum(walls_padded[b_x-1:b_x+1, b_y]) == 2:
        custom_bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = 1
    elif np.sum(walls_padded[b_x, b_y-1:b_y+1]) == 2:
        custom_bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = 1
    else:
        custom_bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = 1
        custom_bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = 1

    features.append(np.sum(custom_bomb_map*crates_padded))

    # Number of agents that explode if bomb is placed at current location (and agent doesn't move)
    for agent in game_state['others']:
        other_agent_pos = agent[3]
        other_agent_pos_x = other_agent_pos[0]
        other_agent_pos_y = other_agent_pos[1]
        agents_in_explosion_radius = 0
        if custom_bomb_map[other_agent_pos_x][other_agent_pos_y] == 1:
            agents_in_explosion_radius += 1
    features.append(agents_in_explosion_radius)

    # Agent straight distance from next bomb (4 if bomb further away than 4) and its cooldown
    bomb_distance = 4
    bomb_cooldown = 4

    for bomb in game_state['bombs']:
        if len(game_state['bombs'])==0:
            break
        else:
            if np.sum(walls_padded[bomb[0][0]:bomb[0][0]+2*PADCONST, bomb[0][1]]) == 2 and bomb[1]==agent_pos_y:
                distance = abs(bomb[0][0] - agent_pos_x)
                cooldown = bomb[1]
                if distance < bomb_distance:
                    bomb_distance = distance
                if cooldown < bomb_cooldown:
                    bomb_cooldown = cooldown
            elif np.sum(walls_padded[bomb[0][0], bomb[0][1]+2*PADCONST:bomb[0][1]]) == 2 and bomb[0]==agent_pos_x:
                distance = abs(bomb[0][1] - agent_pos_y)
                cooldown = bomb[1]
                if distance < bomb_distance:
                    bomb_distance = distance
                if cooldown < bomb_cooldown:
                    bomb_cooldown = cooldown
            elif bomb[0][0]==agent_pos_x:
                distance = abs(bomb[1] - agent_pos_y)
                cooldown = bomb[1]
                if distance < bomb_distance:
                    bomb_distance = distance
                if cooldown < bomb_cooldown:
                    bomb_cooldown = cooldown
            elif bomb[0][1]==agent_pos_y:
                distance = abs(bomb[0][0] - agent_pos_x)   
                cooldown = bomb[1] 
                if distance < bomb_distance:
                    bomb_distance = distance
                if cooldown < bomb_cooldown:
                    bomb_cooldown = cooldown

    features.append(bomb_distance)
    features.append(bomb_cooldown)

    # L1-distance to next agent
    distance_to_next_agent = 30
    closest_agent_x = 0
    closest_agent_y = 0
    for agent in game_state['others']:
        other_agent_pos = agent[3]
        other_agent_pos_x = other_agent_pos[0]
        other_agent_pos_y = other_agent_pos[1]
        distance_btw_agents = abs(other_agent_pos_x - agent_pos_x) + abs(other_agent_pos_y - agent_pos_y)
        if distance_btw_agents < distance_to_next_agent:
            distance_to_next_agent = distance_btw_agents
            closest_agent_x = other_agent_pos_x
            closest_agent_y = other_agent_pos_y
    features.append(distance_to_next_agent)

    # Relative coordinates of next agent
    features.append(closest_agent_x - agent_pos_x)
    features.append(closest_agent_y - agent_pos_y)

    # Coins in radius 3
    coins_in_radius = 0
    for coin in game_state['coins']:
        if coin[0]+PADCONST in range(agent_pos_x_pad - 3, agent_pos_x_pad + 3) and coin[1]+PADCONST in range(agent_pos_y_pad - 3, agent_pos_y + 3):
            coins_in_radius += 1
    features.append(coins_in_radius)


    # L1-distance to next coin
    distance_to_next_coin = 30
    closest_coin_x = 0
    closest_coin_y = 0
    for coin in game_state['coins']:
        coin_x = coin[0]
        coin_y = coin[1]
        distance_to_coins = abs(coin_x - agent_pos_x) + abs(coin_y - agent_pos_y)
        if distance_to_coins < distance_to_next_coin:
            distance_to_next_coin = distance_to_coins
            closest_coin_x = coin_x
            closest_coin_y = coin_y
    features.append(distance_to_next_coin)

    # Relative coordinates of next coin
    features.append(closest_coin_x - agent_pos_x)
    features.append(closest_coin_y - agent_pos_y)

    # Is bomb setting possible?
    bomb_possible = game_state['self'][2]
    features.append(bomb_possible)

    # Agent score
    agent_score = game_state['self'][1]
    features.append(agent_score)

    # Other agents scores
    other_agent_scores = [0, 0, 0]
    for idx, agent in enumerate(game_state['others']):
        other_agent_scores[idx] = agent[1]
    other_agent_scores = np.sort(other_agent_scores)
    for idx, agent_score in enumerate(other_agent_scores):
        features.append(agent_score)

    # Sum of all crates
    crate_sum = np.sum(crates)
    features.append(crate_sum)

    # Crates in radius 1 around next coin
    features.append(np.sum(crates_padded[closest_coin_x+PADCONST-1:closest_coin_x+PADCONST+1, closest_coin_y+PADCONST-1:closest_coin_y+PADCONST+1]))

    # Crates in radius 1 around next agent
    features.append(np.sum(crates_padded[closest_agent_x+PADCONST-1:closest_agent_x+PADCONST+1, closest_agent_y+PADCONST-1:closest_agent_y+PADCONST+1]))

    # Space free of crates in each direction


    # Agent is at left/right border (bool)
    agent_at_leftright_border = 1 if (np.sum(walls_padded[agent_pos_x_pad-3:agent_pos_x_pad,agent_pos_y])==3 or np.sum(walls_padded[agent_pos_x_pad:agent_pos_x_pad+3, agent_pos_y_pad])==3) else 0
    features.append(agent_at_leftright_border)

    # Agent is at top/bottom border (bool)
    agent_at_topbottom_border = 1 if (np.sum(walls_padded[agent_pos_x_pad,agent_pos_y-3:agent_pos_y_pad])==3 or np.sum(walls_padded[agent_pos_x_pad, agent_pos_y_pad:agent_pos_y_pad+3])==3) else 0
    features.append(agent_at_topbottom_border)
    all_features.append(features)


all_features = np.array(all_features)


featurepickle.close()
labelpickle.close()


########################################################################
########################################################################
########################################################################
#########                 FEATURE SELECTION             ################
########################################################################
########################################################################
########################################################################

import pandas as pd
import seaborn as sns
import os 
import scipy.stats as ss
from collections import Counter
import math 
from matplotlib import pyplot as plt
from scipy import stats

# print(all_features.shape)
labels = np.array(labels)
# print(labels.shape)
# all_features = np.concatenate((all_features, labels), axis=0)
# print(all_features)
# print(cols)


noneidx = []
for idx, label in enumerate(labels):
    if label == "UP":
        labels[idx] = 0
    elif label == "DOWN":
        labels[idx] = 1
    elif label == "LEFT":
        labels[idx] = 2
    elif label == "RIGHT":
        labels[idx] = 3
    elif label == "BOMB":
        labels[idx] = 4
    elif label == "WAIT":
        labels[idx] = 5
    else:
        noneidx.append(idx)
        # print("Something went wrong with entry", label)

# print(labels.shape)
labels = np.delete(labels, noneidx, axis=0)
# print(labels.shape)
# print(all_features.shape)
all_features = np.delete(all_features, noneidx, axis=0)
# print(all_features.shape)

X = pd.DataFrame(all_features)
X.columns = cols
# labels = np.array(labels)
y = pd.DataFrame(labels)
y.columns = ["Action"]

# print(y.head)
feature_name = list(X.columns)
num_feats = 12

# print(X.shape)
y = y.stack().astype(float)
# print(X.shape)
# print(type(X))
# print(y.shape)
# print(type(y))
# print(y-y_new)

# print(X.head)

def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        # print("\n", X[i].shape, "\n")
        # print(y.shape, "\n")
        # print(type(X[i]))
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
        
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

# print(cor_feature)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(solver='saga', penalty="l1", max_iter=1000), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
embeded_lgb_selector.fit(X, y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')

# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
print(feature_selection_df.head(num_feats))