
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ## TRAINING Data


train_set = pd.read_csv('../../Data/L2_Enhancement/Data/S2_Hybrid_T1.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values





# ## MODEL

params = {
'n_estimators': range(100,1000,50),
    # 'criterion': ['gini', 'entropy'],
    # 'max_depth': [2,4,12],
    'verbose': [3]

}

# ## TRAINING

#  Grid Search
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

gr = GridSearchCV(estimator=ExtraTreesClassifier(),param_grid=params, scoring= 'accuracy', cv=10, n_jobs=-1)

# fitting and extracting the best parameters
gr.fit(featureMatrix, labelVector)
best_parameters = gr.best_params_
print("Best Params: ",best_parameters)
best_result = gr.best_score_
print("Best Results: ",best_result)


# Detailed results of the best performing model

model= ExtraTreesClassifier(n_estimators=best_parameters["n_estimators"])

# Getting the feature importance

# getting the features

features_list = list(train_set.columns)

features_list.remove('class')

# fitting the model

# XGB DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 200)

model.fit(X_train, y_train)




import shap

shap.initjs()

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(featureMatrix)

# visualize the first prediction's explanation

features_list = list(train_set.columns)
features_list.remove('class')

# shap.summary_plot(shap_values[1], featureMatrix, feature_names = features_list, max_display=20,plot_size= [20,12], )
# 2
shap.summary_plot(shap_values, featureMatrix, plot_type="bar",max_display=50, class_names= [0,1,2,3,4], plot_size= [15,15], feature_names = features_list,
                  )



# Class 0

shap.summary_plot(shap_values[0], featureMatrix,feature_names = features_list, max_display=20,plot_size= [17,10], show=False)
# Workaround because of title not showing
plt.title("Class 0: lminalA prediction")
plt.show()
# Class 1

shap.summary_plot(shap_values[1], featureMatrix,feature_names = features_list, max_display=20,plot_size= [17,10],show=False )
plt.title("Class 1: lminalB prediction")
plt.show()
# Class 2

shap.summary_plot(shap_values[2], featureMatrix,feature_names = features_list, max_display=20,plot_size= [17,10],show=False )
plt.title("Class 2: TNBC prediction")
plt.show()
# Class 3

shap.summary_plot(shap_values[3], featureMatrix,feature_names = features_list, max_display=20,plot_size= [17,10],show=False )
plt.title("Class 3: ERBB prediction")
plt.show()
# Class 4

shap.summary_plot(shap_values[4], featureMatrix,feature_names = features_list, max_display=20,plot_size= [17,10],show=False )
plt.title("Class 4: normal prediction")
plt.show()


# hmexp = shap.Explainer(model, featureMatrix)
# hmsv = hmexp(featureMatrix[:606])
# shap.plots.heatmap(shap_values, feature_values=hmsv.abs.max(0))

# shap_values = shap.TreeExplainer(model).shap_values(featureMatrix)
# shap.summary_plot(shap_values, X_train, plot_type="bar")

# https://medium.com/mlearning-ai/shap-force-plots-for-classification-d30be430e195



# XGBoost Feature importance
import seaborn as sns
feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = features_list
feature_importance_df["importance"] = model.feature_importances_

feature_importance_df  = feature_importance_df.sort_values('importance', ascending=False,ignore_index=True)
plt.figure(figsize = (15, 10))
sns.barplot(x ='importance', y ='feature', data= feature_importance_df[:50],palette ="rocket")
sns.color_palette("rocket")
plt.show()


# calculating Cummulative

DNA_N = 0
DNA_Cumm = 0
RNA_N = 0
RNA_Cumm = 0
CNV_N = 0
CNV_Cumm = 0

top50 = feature_importance_df[:50]



import re
regexpD=re.compile(r'_DNA$')
regexpR=re.compile(r'_RNA$')
regexpC=re.compile(r'_CNV$')




# T1
for feature in train_set.columns:
    if regexpD.search(feature):
        DNA_N += 1

    elif regexpR.search(feature):
        RNA_N += 1

    elif regexpC.search(feature):
        CNV_N += 1

print("T1 \nDNA: ",DNA_N)
print("RNA: ", RNA_N)
print("CNV: ", CNV_N)

# Top 50

DNA_N = 0
DNA_Cumm = 0
RNA_N = 0
RNA_Cumm = 0
CNV_N = 0
CNV_Cumm = 0
for index, row in top50.iterrows():
    if regexpD.search(str(row['feature'])):
        DNA_N += 1

    elif regexpR.search(str(row['feature'])):
        RNA_N += 1

    elif regexpC.search(str(row['feature'])):
        CNV_N += 1

print("TOP 50 \nDNA: ",DNA_N)
print("RNA: ", RNA_N)
print("CNV: ", CNV_N)

DNA_N = 0
DNA_Cumm = 0
RNA_N = 0
RNA_Cumm = 0
CNV_N = 0
CNV_Cumm = 0

for index, row in feature_importance_df.iterrows():

    if  regexpD.search(str(row['feature'])):

        DNA_N+=1
        DNA_Cumm+=row['importance']
    elif  regexpR.search(str(row['feature'])):

        RNA_N+=1
        RNA_Cumm+=row['importance']
    elif  regexpC.search(str(row['feature'])):

        CNV_N+=1
        CNV_Cumm+=row['importance']


print("All \n  DNA: ",DNA_N,", Total importance",DNA_Cumm )
print("RNA: ",RNA_N,", Total importance",RNA_Cumm )
print("CNV: ",CNV_N,", Total importance",CNV_Cumm )




# Feature correlation

plt.figure(figsize = (10, 8))
cor = np.corrcoef(featureMatrix)
# sns.heatmap(cor,annot=True, cmap=plt.cm.CMRmap_r)
# plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class_names = ['lminalA', 'lminalB', 'TNBC', 'ERBB','Unclear']
class_names2 = [0,1,2,3,4]
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=class_names2,normalize='all')

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names,)

disp.plot(cmap='cividis', xticks_rotation='vertical',)
plt.rcParams["figure.figsize"] = (12,12)
plt.show()