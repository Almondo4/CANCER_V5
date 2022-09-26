
# TODO: check if you are extrating the features according to their respective importances
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

def calculateResults2 (optimalModel, featureMatrix,labelVector):
    results = cross_validate(optimalModel, featureMatrix, labelVector, scoring=('accuracy', 'precision_weighted',
                                                                                'f1_weighted', 'recall_weighted',
                                                                                'roc_auc_ovo_weighted'), cv=10,
                             verbose=0, n_jobs=-1, return_estimator=True)

    return results

# ## TRAINING Data

train_set = pd.read_csv('../../Baseline/B_HYBRID.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values
features = list(train_set.columns)


# removing the indexes and labels from features
features.remove('Class')


# Extracting feaaure Importance

# ## MODEL

import xgboost as xgb

# XGB DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 0)
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
    "max_depth":1000,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 5,
    "verbosity": 1}
epochs = 20
model = xgb.XGBClassifier(**param)

# # initial results inspection
# from Results import calculateResults
#
# calculateResults(model,featureMatrix,labelVector)

# Feature importance time
model.fit(featureMatrix,labelVector)

thresholds = np.sort(model.feature_importances_)

i = thresholds.mean()



# Extrating the features vector
XGB_selected_Features = []
mask = [x for x in thresholds if x != 0]
features_importance=zip(mask, features)
for bool, feature in zip(mask, features):
    if bool:
        XGB_selected_Features.append(feature)

print('XGV Best Features: ', XGB_selected_Features)



# Creating new dataset

# labels
Class_labels= train_set.pop('Class')


new_dataset = train_set[XGB_selected_Features]

new_dataset.columns = XGB_selected_Features

new_dataset= new_dataset.assign(Class=Class_labels)


new_dataset.to_csv('S1_hybrid_XGB_T2.csv',index=False)

# # select the features
# from sklearn.feature_selection import SelectFromModel
# selection = SelectFromModel(model, threshold=thresh, prefit=True)
# selection = SelectFromModel(model, threshold=thresholds[2], prefit=True)
#
# feature_idx = selection.get_support()
# print(feature_idx)
# # array([ True,  True,  True, False, False])
#
# selected_dataset = selection.transform(X_test)
# print(selected_dataset.shape)
# # (200, 3)
#
#  #  Creating new datase4t from new array
#
#     # 1. creating dataframe of Feature importance
#
# selected_XGB_Hybrid_Features = pd.DataFrame({'feature': features,
#                                           'Ranking':feature_importances_})
#
#
#     # Creating New Dataset from selected Features
# new_featureMatrix = x.transform(featureMatrix)
#
# newFrame = np.insert(new_featureMatrix,new_featureMatrix[0].__len__(),labelVector,axis=1) # you should check if you keep getting the same number of features
# Hybrid_RFE = pd.DataFrame(newFrame)
#
# rfe_selected_Features.append('class')
# Hybrid_RFE.columns = rfe_selected_Features





