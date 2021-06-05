# -*- coding: utf-8 -*-
# @Time:2021/4/1316:22
# @File:Number_chain_ring.py
import pandas as pd
import numpy as np
df=pd.read_excel("C:/Users/97899/Desktop/N/Network/Strong_index.xls")
count=0
count1=0
print(np.shape(df))
for i in range(np.shape(df)[0]):
    for j in range(1,np.shape(df)[1]):
        if df.iloc[i,j]>=0:
            count+=1
            if df.iloc[i,j]>0:
                count1+=1
print(count,count1)
count=0
count1=0
for year in range(2008,2021):
    n=2
    df_1=pd.read_excel("C:/Users/97899/Desktop/N/Network/circle21.xls",sheet_name=str(year))
    for i in df_1.loc[:,3]:
        if i >=0:
            count += 1
            if i > 0:
                count1 += 1
        # if i >0 df.iloc[i,2]:

print(count,count1)


