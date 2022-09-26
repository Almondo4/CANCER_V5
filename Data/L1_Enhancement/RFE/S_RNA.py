import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# Data
RNA_DATA = pd.read_csv('../../Baseline/B_RNA.csv')
# Hybrid_features= pd.read_csv('../../Data/Features/Hybrid_Features.csv').values



labelVector = RNA_DATA.iloc[:, -1].values
        # featureMatrix
featureMatrix = RNA_DATA.iloc[:, :-1].values

features = list(RNA_DATA.columns)


# removing the indexes and labels from features
features.remove('Class')



    # 2.    Extracting using RFE

from sklearn.feature_selection import RFECV

Eclf = RandomForestClassifier(n_estimators=2000)
print('Now trying with RFE: \n')
rfecv = RFECV(estimator= Eclf , step= 200, cv =3 , scoring='accuracy',verbose=2,n_jobs= -1,)
rfecv = rfecv.fit(featureMatrix,labelVector)

# extracting the number of features
print('RFE Optimal Number of Features: ', rfecv.n_features_)
#
# selected_X_train = rfecv.transform(X_train)
# selected_X_train = rfecv.transform(X_test)


# Extrating the features vector
rfe_selected_Features = []
mask = rfecv.get_support()

for bool, feature in zip(mask, features):
    if bool:
        rfe_selected_Features.append(feature)

print('RFE Best Features: ', rfe_selected_Features)

 #  Creating new datase4t from new array

    # 1. creating dataframe of Feature importance

selected_RFE_RNA_Features = pd.DataFrame({'feature': features,
                                          'Ranking':rfecv.ranking_})


    # Creating New Dataset from selected Features
new_featureMatrix = rfecv.transform(featureMatrix)

newFrame = np.insert(new_featureMatrix,new_featureMatrix[0].__len__(),labelVector,axis=1) # you should check if you keep getting the same number of features
RNA_RFE = pd.DataFrame(newFrame)

rfe_selected_Features.append('class')
RNA_RFE.columns = rfe_selected_Features



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
    "verbosity": 1}
epochs = 20
model = xgb.XGBClassifier(**param)

print('Testing Results with New DataSet')
from Results import calculateResults
calculateResults(optimalModel=model,labelVector= labelVector,featureMatrix= new_featureMatrix)
#   Saving Results


RNA_RFE.to_csv('S_RNA.csv',index=False) #<--------------------------------

selected_RFE_RNA_Features.to_csv('S1_RNA_FeatureRanking.csv')
