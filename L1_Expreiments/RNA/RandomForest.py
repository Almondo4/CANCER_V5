import pandas as pd

# ## TRAINING Data

train_set = pd.read_csv('../../Data/L1_Enhancement/S_RNA.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values


# ## MODEL
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


params = {
'n_estimators': range(100,1000,50),
    # 'criterion': ['gini', 'entropy'],
    # 'max_depth': [2,4,12],
    'verbose': [3]

}

# ## TRAINING

#  Grid Search

from sklearn.model_selection import GridSearchCV

gr = GridSearchCV(estimator=model,param_grid=params, scoring= 'accuracy', cv=10, n_jobs=-1)

# fitting and extracting the best parameters
gr.fit(featureMatrix, labelVector)
best_parameters = gr.best_params_
print("Best Params: ",best_parameters)
best_result = gr.best_score_
print("Best Results: ",best_result)


# Detailed results of the best performing model

optimalModel= RandomForestClassifier(n_estimators=best_parameters["n_estimators"])

# ## TESTING

# Results

from Results import calculateResults

calculateResults(optimalModel, featureMatrix, labelVector)

