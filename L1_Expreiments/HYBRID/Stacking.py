
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('../../Data/L1_Enhancement/S1_Hybrid_os_T1.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values


# ## MODEL
# Initilazing the classifier


from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier



estimators = [
    ('rf', RandomForestClassifier(n_estimators=400)),
    # ('et',  ExtraTreesClassifier()),
    # ('ada',AdaBoostClassifier(n_estimators=100)),
    ('gb',GradientBoostingClassifier(n_estimators=100, learning_rate=0.8, max_depth=1000, )),
 ]

# learning the voting scheme
#
# classifier = StackingClassifier(
#     estimators=estimators, final_estimator=LogisticRegression()
# )

# from sklearn.ensemble import VotingClassifier
#
# classifier = VotingClassifier(estimators=estimators,
#                         voting='soft',)


# Results

from Results import calculateResults

calculateResults(classifier,featureMatrix,labelVector)
