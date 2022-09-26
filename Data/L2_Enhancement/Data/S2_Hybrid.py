import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# Data
Hybrid_DATA = pd.read_csv('../../L1_Enhancement/S1_Hybrid_os_T1.csv')

labelVector = Hybrid_DATA.iloc[:, -1].values
        # featureMatrix
featureMatrix = Hybrid_DATA.iloc[:, :-1].values

features = list(Hybrid_DATA.columns)


# removing the indexes and labels from features
features.remove('class')



    # 2.    Extracting using RFE

from sklearn.feature_selection import RFECV

Eclf = RandomForestClassifier(n_estimators=100)
print('Now trying with RFE: \n')
rfecv = RFECV(estimator= Eclf , step= 1, cv =5 , scoring='accuracy',verbose=2,n_jobs= -1,)
rfecv = rfecv.fit(featureMatrix,labelVector)


print('RFE Optimal Number of Features: ', rfecv.n_features_)



# Extrating the features vector
rfe_selected_Features = []
mask = rfecv.get_support()

for bool, feature in zip(mask, features):
    if bool:
        rfe_selected_Features.append(feature)

print('RFE Best Features: ', rfe_selected_Features)

 #  Creating new datase4t from new array

    # 1. creating dataframe of Feature importance

selected_RFE_Hybrid_Features = pd.DataFrame({'feature': features,
                                          'Ranking':rfecv.ranking_})


    # Creating New Dataset from selected Features
new_featureMatrix = rfecv.transform(featureMatrix)

newFrame = np.insert(new_featureMatrix,new_featureMatrix[0].__len__(),labelVector,axis=1) # you should check if you keep getting the same number of features
Hybrid_RFE = pd.DataFrame(newFrame)

rfe_selected_Features.append('class')
Hybrid_RFE.columns = rfe_selected_Features



# testing again with new dataset

import xgboost as xgb

# XGB DATA
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 0)
# train = xgb.DMatrix(X_train, label=y_train)
# test = xgb.DMatrix(X_test, label=y_test)

param = {
    "max_depth":1000,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 5,
    "verbosity": 3}
epochs = 20
model = xgb.XGBClassifier(**param)

print('Testing Results with New DataSet')
from Results import calculateResults
calculateResults(optimalModel=model,labelVector= labelVector,featureMatrix= new_featureMatrix)

#   Saving Results


Hybrid_RFE.to_csv('S2_Hybrid_T1.csv', index=False) #<--------------------------------

selected_RFE_Hybrid_Features.to_csv('S2_Hybrid_FeatureRanking_T1.csv')
