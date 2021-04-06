# -*- coding: utf-8 -*-
# @Time:2021/4/615:35
# @File:Fig_N_NODF.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path="C:/Users/97899/Desktop/N/Network/loop_NODF.xls"
df_NODF=pd.read_excel(path)
gb=df_NODF.groupby(["Fre"])
NODF=[]
com=[]
for g in gb:
    n=np.shape(g[1])[0]
    if int(g[0])==0:
        g_all = g[1]["NODF"].values.tolist() + np.repeat([0], 26 - n).tolist()
        c_all = g[1]["complexity"].values.tolist() + np.repeat([0], 26 - n).tolist()
        NODF.append(np.mean(g_all))
        com.append(np.mean(c_all))
    else:
        if n < 234:#N=52, M=234
            g_all=g[1]["NODF"].values.tolist()+np.repeat([0],234-n).tolist()
            c_all=g[1]["complexity"].values.tolist()+np.repeat([0],234-n).tolist()
            print(g_all)
            NODF.append(np.mean(g_all))
            com.append(np.mean(c_all))

print(NODF)
N=("0","1","2","3","5","10","15","20","50")
F=("0","2","12")
M=("No","Yes")
# print(N)
bar_width=0.3
x1=np.arange(3)
x2=x1+bar_width
plt.bar(x1,NODF,width=bar_width,color="b",label="NODF")
plt.bar(x2, com,width=bar_width,color="g",label="Complexoty")
plt.xticks(x1+bar_width/2,F)
plt.xlabel("Frequency")
#N Addition Rate
plt.ylabel("Nested values")
plt.legend()
plt.show()