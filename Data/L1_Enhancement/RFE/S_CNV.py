import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# Data
CNV_DATA = pd.read_csv('../../Baseline/B_CNV.csv')
# Hybrid_features= pd.read_csv('../../Data/Features/Hybrid_Features.csv').values



labelVector = CNV_DATA.iloc[:, -1].values
        # featureMatrix
featureMatrix = CNV_DATA.iloc[:, :-1].values

features = list(CNV_DATA.columns)


# removing the indexes and labels from features
features.remove('Class')



    # 2.    Extracting using RFE

from sklearn.feature_selection import RFECV

Eclf = RandomForestClassifier(n_estimators=1000)
print('Now trying with RFE: \n')
rfecv = RFECV(estimator= Eclf , step= 100, cv =3 , scoring='accuracy',verbose=2,n_jobs= -1,)
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

selected_RFE_CNV_Features = pd.DataFrame({'feature': features,
                                          'Ranking':rfecv.ranking_})


    # Creating New Dataset from selected Features
new_featureMatrix = rfecv.transform(featureMatrix)

newFrame = np.insert(new_featureMatrix,new_featureMatrix[0].__len__(),labelVector,axis=1) # you should check if you keep getting the same number of features
CNV_RFE = pd.DataFrame(newFrame)

rfe_selected_Features.append('class')
CNV_RFE.columns = rfe_selected_Features



# testing again with new dataset





#   Saving Results


CNV_RFE.to_csv('S_CNV.csv', index=False) #<--------------------------------

selected_RFE_CNV_Features.to_csv('S1_CNV_FeatureRanking.csv')
