# -*- coding: utf-8 -*-
# @Time:2021/4/1616:15
# @File:Fig6_Enveriment_dist.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import statsmodels.api as sm



def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""环境标准化"""


def get_normalise(lst_chara):
    re_lst=lst_chara
    for z in re_lst:
        z_m = np.mean(z)
        z_sigma = np.std(z)
        z=[(zi-z_m)/z_sigma for zi in z]

    return re_lst

"""获得欧氏距离"""

def get_Euclidean(lst_chara):
    dist = []
    data = get_normalise(lst_chara)
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            ed=np.sum((np.array(data[i]) - np.array(data[j])) ** 2)
            dist.append(np.sqrt(ed))

    return dist



if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_NODF = pd.read_excel(path + "Network/loop_NODF.xls")
    dic_byrank = LoadDict(path + "Attribute/groupbyrank.txt")
    dic_pairdist=dict([(key,[]) for key in dic_byrank.keys()])
    for key in dic_byrank.keys():
        envir=[]
        for item in dic_byrank[key]:
            year=item[0]
            ex=item[1]
            if year==2008.0:
                envir.append([0,0,0])
            else:
                df_year=df_NODF[df_NODF["year"]==year]
                df_ex=df_year[df_year["ex"]==ex]
                N=df_ex["N"].values
                M=df_ex["Fre"].values
                F=df_ex["Mow"].values
                rain=df_ex["rain"].values
                temp=df_ex["temper"].values
                envir.append([N,M,F])
        dic_pairdist[key]=get_Euclidean(envir)

    pairdist_mean=[]
    pairdist_sd=[]
    for key in dic_pairdist.keys():
        pairdist_mean.append(np.mean(dic_pairdist[key]))
        pairdist_sd.append(np.std(dic_pairdist[key]))

    """画图"""
    # labels = [">0.85", "0.85-0.8", "0.8-0.75", "0.75-0.7"]
    labels=[">0.85","0.85-0.8","0.8-0.75","0.75-0.7","0.7-0.65","0.65-0.6"]
    # labels=["0.6-0.55", "0.55-0.5", "0.5-0.45", "0.45-0.4", "0.4-0.35", "0.35>"]
    width = 0.35
    x1=[1,2,3,4,5,6]
    x2 = [[i, i ** 2] for i in x1]
    x = sm.add_constant(x1)
    model = sm.OLS(pairdist_mean[:6], x)
    results = model.fit()
    print(results.summary())
    y_fitted = results.fittedvalues

    plt.plot(x1, y_fitted, "r")
    plt.ylim(6, 14)
    plt.bar(x1, pairdist_mean[:6], width=0.35,)
    plt.ylabel("Environment Euclidean", fontdict={"size": 15})
    plt.xlabel("Mean Competition-Coefficient(Superior)", fontdict={"size": 15})
    plt.xticks(x1, labels)
    plt.text(2,13,r"$r^2=0.410,P=0.171$")

    plt.show()