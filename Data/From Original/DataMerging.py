# concatenate 2 numpy arrays: column-wise
# >np.concatenate((array2D_1,array2D_2),axis=1)
# array([[ 0,  1,  2, 10, 11, 12],
#        [ 3,  4,  5, 13, 14, 15],
#        [ 6,  7,  8, 16, 17, 18]])


import pandas as pd
import numpy as np


RNA_DATA = pd.read_csv('RNA_DATA.csv')
DNA_DATA = pd.read_csv('DNA_DATA.csv')
CNV_DATA = pd.read_csv('CNV_DATA.csv')



    # TODO: encode the data first
        # Vector of labels
labelVector =  DNA_DATA.iloc[:,-1].values
        # featureMatrix
featureMatrixDNA = DNA_DATA.iloc[:,:-1].values
featureMatrixRNA = RNA_DATA.iloc[:,:-1].values
featureMatrixCNV = CNV_DATA.iloc[:,:-1].values

# Creating the Hybrid Feature Matrix
Hybrid = np.hstack((featureMatrixDNA,featureMatrixRNA,featureMatrixCNV))
HybridFrame = np.insert(Hybrid,42666,labelVector,axis=1)
HybridFrame = pd.DataFrame(HybridFrame)
# Creating the Feature labels vector

DNA_features = DNA_DATA.columns
RNA_features = RNA_DATA.columns
CNV_features = CNV_DATA.columns


# testing whether the vectors are uniq

DNA_features_list = list(DNA_features)
RNA_features_list = list(RNA_features)
CNV_features_list = list(CNV_features)
DNA_features_list.remove('Class')
RNA_features_list.remove('Class')
CNV_features_list.remove('Class')


def uniqTest(DNA_features_list,RNA_features_list,CNV_features_list):
        # DNA in RNA
    DiR = 0
    try:
        for i in range(DNA_features_list.__len__()):
            if DNA_features_list[i] == RNA_features_list[i]:
                DiR+=1
        print(f"There are: {DiR} DNA features in the RNA Dataset",)
    except Exception as e:
        print(f"There are: {DiR} DNA features in the RNA Dataset .... + Exception: {e}")


        # DNA in CNV
    DiC = 0
    try:
         for  i in range(DNA_features_list.__len__()):
            if DNA_features_list[i] == CNV_features_list[i]:
                DiC += 1
         print(f"There are: {DiC} DNA features in the CNV Dataset")
    except Exception as e:
         print(f"Exception: probably the number is higher.... + {e}")


        # RNA in CNV
    RiC = 0
    try:
        for i in range(RNA_features_list.__len__()):
            if RNA_features_list[i] == CNV_features_list[i]:
                RiC += 1
        print(f"There are: {RiC} DNA features in the RNA Dataset")
    except Exception as e:
        print(f"Exception: probably the number is higher.... + {e}")


uniqTest(DNA_features_list,RNA_features_list,CNV_features_list)
# renaming eveyfeature according to its dataset:

    # DNA
for i in range(DNA_features_list.__len__()):
    DNA_features_list[i]=DNA_features_list[i]+"_DNA"

    # RNA
for i in range(RNA_features_list.__len__()):
    RNA_features_list[i]=RNA_features_list[i]+"_RNA"

    # CNV
for i in range(CNV_features_list.__len__()):
    CNV_features_list[i]=CNV_features_list[i]+"_CNV"

# saving the feature vectors
# np.savetxt('DNA_Features.csv', DNA_features_list, delimiter=',', fmt='%s')
# np.savetxt('RNA_Features.csv', RNA_features_list, delimiter=',', fmt='%s')
# np.savetxt('CNV_Features.csv', CNV_features_list, delimiter=',', fmt='%s')

#  Merging the vectors

mergedLabels = np.hstack((DNA_features_list,RNA_features_list,CNV_features_list,['Class']))
HybridFrame.columns = mergedLabels
HybridFrame.to_csv("./Hybrid_DATA.csv")

#
#
# # Transforming into a dataFrame
# CNV_DATA_New = pd.DataFrame(HybridFrame)
# CNV_DATA_New.to_csv("./Hybrid_DATA.csv")


#
# # Original Dataset to extract the features from
# DNA = pd.read_csv("brca_methDNA_subtype.csv")
# # Extracting Features
# DNA_features = list(DNA['Features'])
# DNA_features.append('Class')
# # Altering DATA v1
# DNA_New= pd.read_csv('../DNA_DATA.csv')
# # feature matrix without sequential indexes
# DNA_FM= DNA_New.iloc[:,1:-1].values
# # labels
# DNA_NEW_Labels = DNA_New.iloc[:,-1].values
#
#
# # Creating V2 from new Feature Matrix
# DNA_DATA_v2 = np.insert(DNA_FM, 14285, DNA_NEW_Labels, axis=1)
# # Transforming into a dataFrame
# DNA_DATA_v2 = pd.DataFrame(DNA_DATA_v2)
# # naming The features
# DNA_DATA_v2.columns = DNA_features
# DNA_DATA_v2.to_csv("../DNA_DATAv2.csv")