
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ## TRAINING Data

T1 = pd.read_csv('../../Data/L1_Enhancement/S1_Hybrid_os_T1.csv')

T2 = pd.read_csv('../../Data/L1_Enhancement/S1_Hybrid_T2.csv')




DNA_N = 0
DNA_Cumm = 0
RNA_N = 0
RNA_Cumm = 0
CNV_N = 0
CNV_Cumm = 0



import re
regexpD=re.compile(r'_DNA$')
regexpR=re.compile(r'_RNA$')
regexpC=re.compile(r'_CNV$')

# T1
for feature in T1.columns:
    if regexpD.search(feature):
        DNA_N += 1

    elif regexpR.search(feature):
        RNA_N += 1

    elif regexpC.search(feature):
        CNV_N += 1

print("T1 \nDNA: ",DNA_N)
print("RNA: ", RNA_N)
print("CNV: ", CNV_N)

# T2


DNA_N = 0
DNA_Cumm = 0
RNA_N = 0
RNA_Cumm = 0
CNV_N = 0
CNV_Cumm = 0

for feature in T2.columns:
    if regexpD.search(feature):
        DNA_N += 1

    elif regexpR.search(feature):
        RNA_N += 1

    elif regexpC.search(feature):
        CNV_N += 1

print("T2 \nDNA: ",DNA_N)
print("RNA: ", RNA_N)
print("CNV: ", CNV_N)