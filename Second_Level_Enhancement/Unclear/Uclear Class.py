
import pandas as pd
import numpy as np

L1_hybrid = pd.read_csv('../../Data/L2_Enhancement/Data/S2_Hybrid_T1.csv')


unclear = L1_hybrid[ L1_hybrid['class'] == 4 ]

unclear.to_csv('unclear.csv',index=False) # < ----------------------------------

