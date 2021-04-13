# -*- coding: utf-8 -*-
# @Time:2021/4/1319:41
# @File:Ellipse_95.py
"""计算椭圆百分之95置信区间内的情况占比"""

import pandas as pd
import numpy as np

path = "C:/Users/97899/Desktop/N/"
df_ICN = pd.read_excel(path + "ICN.xls")
# df_TCN=pd.read_excel(path+"TCN.xls")
dic_ICN = {}
dic_ICN["low-NM"] = 0
dic_ICN["low-M"] = 0
dic_ICN["high-NM"] = 0
dic_ICN["high-M"] = 0

for i in range(np.shape(df_ICN)[0]):
    print(int(df_ICN.iloc[i, 4]),int(df_ICN.iloc[i, 6]))
    if int(df_ICN.iloc[i, 4]) == 0 and int(df_ICN.iloc[i, 6]) == 0:
        dic_ICN["low-NM"] += 1
    if int(df_ICN.iloc[i, 4]) == 1 and int(df_ICN.iloc[i, 6]) == 0:
        dic_ICN["low-M"] += 1
    if int(df_ICN.iloc[i, 4]) == 0 and int(df_ICN.iloc[i, 6]) == 1:
        dic_ICN["high-NM"] += 1
        print("有")
    if int(df_ICN.iloc[i, 4]) == 1 and int(df_ICN.iloc[i, 6]) == 1:
        dic_ICN["high-M"] += 1
print(dic_ICN)
dic_ICN["low-NM"] = dic_ICN["low-NM"] / np.shape(df_ICN)[0]
dic_ICN["low-M"] = dic_ICN["low-M"] / np.shape(df_ICN)[0]
dic_ICN["high-NM"] = dic_ICN["high-NM"] / np.shape(df_ICN)[0]
dic_ICN["high-M"] = dic_ICN["high-M"] / np.shape(df_ICN)[0]
print(dic_ICN)
