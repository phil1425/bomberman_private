########################################################################
########################################################################
########################################################################
#########                 FEATURE SELECTION             ################
########################################################################
########################################################################
########################################################################
import numpy as np
import pickle
from feature_extraction import FeatureSelector

import pandas as pd
import seaborn as sns
import os 
import scipy.stats as ss
from collections import Counter
import math 
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from scipy import stats

def plot_settings(title, xlabel, ylabel):
    
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)



featurepickle = open("./featureset_large.pt", "rb")
labelpickle = open("./labelset_large.pt", "rb")

gamestates = pickle.load(featurepickle)
labels = pickle.load(labelpickle)

# all_features = []
cols = ["Step Number", "Sum of crates radius 1", "Sum of crates radius 2", #"Agent in explosion zone",
        "Crates explode if bomb here", "Agents explode if bomb here", #"Straight bomb distance", "Straight bomb cooldown",
        "Distance next agent", "Next agent rel x", "Next agent rel y", "Crates around next agent", "Agent x", "Agent y", "Coins in radius 3", "Distance next coin", "Next coin rel x",
        "Next coin rel y", "Crates around next coin", "Bomb possible", "Agent score", "Other Agent 1 score", "Other Agent 2 score", "Other Agent 3 score",
        "Sum all crates", "Agent at left/right border", "Agent at top/bottom border", "Agent pos rel to wall", "Free space down", "Free space right", "Free space up", "Free space left",
        "Survivable spaces down", "Survivable spaces right", "Survivable spaces up", "Survivable spaces left", "Survivable spaces wait"]

#####################################
#   call feature extractor here     #
#####################################

selector = FeatureSelector()

# print(len(gamestates))
all_features = np.zeros((len(gamestates), len(cols)))
# print(all_features.shape)

for idx, game_state in enumerate(gamestates):
    features = selector.eval_all_features(game_state)
    print("#####################################")
    print(f"Extracting Feature {idx}")
    print("#####################################\n\n")
    all_features[idx] = features



featurepickle.close()
labelpickle.close()






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
num_feats = 25

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
        
    return cor_support, cor_feature, cor_list



cor_support, cor_feature, cor_list = cor_selector(X, y, num_feats)
feature_strength = np.sort(np.abs(cor_list))[-num_feats:]
print(str(len(cor_feature)), 'selected features')

plot_settings("Pearson Correlation with next Action", "Features", "Correlation Strength")
plt.plot([idx for idx, feature in enumerate(cor_feature)], feature_strength, "x", label = 'Pearson')
plt.savefig("./pearson.pdf")
# plt.show()
# print(cor_feature)
print(cor_feature[-8:])

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
# chi_selector = SelectKBest(chi2, k=num_feats)
# chi_selector.fit(X_norm, y)
# chi_support = chi_selector.get_support()
# chi_feature = X.loc[:,chi_support].columns.tolist()
# print(str(len(chi_feature)), 'selected features')

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_params = rfe_selector.get_params()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

# print(rfe_params)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embedded_lr_selector = SelectFromModel(LogisticRegression(solver='saga', penalty="l1", max_iter=1000), max_features=num_feats)
embedded_lr_selector.fit(X_norm, y)

embedded_lr_support = embedded_lr_selector.get_support()
embedded_lr_params = embedded_lr_selector.get_params()
embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
print(str(len(embedded_lr_feature)), 'selected features')

# print(embedded_lr_params)

# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

# embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
# embedded_rf_selector.fit(X, y)

# embedded_rf_support = embedded_rf_selector.get_support()
# embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
# print(str(len(embedded_rf_feature)), 'selected features')


# from sklearn.feature_selection import SelectFromModel
# from lightgbm import LGBMClassifier

# lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
#             reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

# embedded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
# embedded_lgb_selector.fit(X, y)

# embedded_lgb_support = embedded_lgb_selector.get_support()
# embedded_lgb_feature = X.loc[:,embedded_lgb_support].columns.tolist()
# print(str(len(embedded_lgb_feature)), 'selected features')

# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 
                                    'Pearson':cor_support, 
                                    #'Chi-2':chi_support, 
                                    'RFE':rfe_support, 
                                    'Lasso':embedded_lr_support})
                                    #'Random Forest':embedded_rf_support, 
                                    #'LightGBM':embedded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
print(feature_selection_df.head(num_feats))