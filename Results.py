from sklearn.model_selection import cross_validate
import numpy as np


def calculateResults (optimalModel, featureMatrix,labelVector):
    results = cross_validate(optimalModel, featureMatrix, labelVector, scoring=('accuracy', 'precision_weighted',
                                                                                'f1_weighted', 'recall_weighted',
                                                                                'roc_auc_ovo_weighted'), cv=10,
                             verbose=0, n_jobs=-1, return_estimator=True)

                #Extracting the results for later manipulation

    accuracy= results['test_accuracy']
    precision= results['test_precision_weighted']
    recall= results['test_recall_weighted']
    f1= results['test_f1_weighted']
    roc= results['test_roc_auc_ovo_weighted']
    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("f1 :", f1)
    print("Roc_AUC :", roc)

                #get the mean of each  result

    print("++++++Mean Resualts+++++")

    print("Accuracy: ",accuracy.mean() * 100)
    print("Pricision: ",precision.mean() * 100)
    print("Recall: ",recall.mean() * 100)
    print("F1: ",f1.mean() * 100)
    print("Roc_Auc: ",roc.mean() * 100)

def calculate(m, f,l,fts):
    results = cross_validate(m, f, l, scoring=('accuracy','precision_weighted',
                                                                                    'f1_weighted','recall_weighted',
                                                                                    'roc_auc_ovo_weighted'), cv = 3,
                                 verbose=0, n_jobs=-1, return_estimator=True)

                    #Extracting the results for later manipulation

    accuracy= results['test_accuracy']
    precision= results['test_precision_weighted']
    recall= results['test_recall_weighted']
    f1= results['test_f1_weighted']
    roc= results['test_roc_auc_ovo_weighted']
    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("f1 :", f1)
    print("Roc_AUC :", roc)

                    #get the mean of each  result

    print("++++++Mean Resualts+++++")
    print("Accuracy: ",accuracy.mean() * 100)
    print("Pricision: ",precision.mean() * 100)
    print("Recall: ",recall.mean() * 100)
    print("F1: ",f1.mean() * 100)
    print("Roc_Auc: ",roc.mean() * 100)
    model = results['estimator'][0]

    # meanTH = np.sort(model.feature_importances_)
    # meanTH = map(lambda x: x *10000, meanTH)
    # meanTH = np.median(meanTH)

    featureImp = list(zip(fts,model.feature_importances_))
    np.savetxt('ETC_featureImp_DNA.csv', featureImp, delimiter=',', fmt='%s')
    return model


def importance(fImportance):
    import re
    regexpD = re.compile(r'DNA')
    regexpR = re.compile(r'RNA')
    regexpC = re.compile(r'CNV')
    importance = {'DNA': {'imp': 0, "n_feature": 0}, 'RNA': {'imp': 0, "n_feature": 0},
                  'CNV': {'imp': 0, "n_feature": 0}}

    for f in fImportance:
        if regexpD.search(f[0]):
            importance['DNA']['imp'] += f[1]
            importance['DNA']['n_feature'] += 1
            continue
        elif regexpR.search(f[0]):
            importance['RNA']['imp'] += f[1]
            importance['RNA']['n_feature'] += 1
            continue
        elif regexpC.search(f[0]):
            importance['CNV']['imp'] += f[1]
            importance['CNV']['n_feature'] += 1
            continue
    print("Importance: ", importance)
    return importance
