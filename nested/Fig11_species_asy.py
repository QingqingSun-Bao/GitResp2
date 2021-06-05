# -*- coding: utf-8 -*-
# @Time:2021/5/1815:09
# @File:Fig11_species_asy.py
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline

if __name__ == "__main__":
    df_comp = pd.read_excel("C:/Users/97899/Desktop/N/Network/Strong_index.xls")
    df_ex = pd.read_excel("C:/Users/97899/Desktop/N/实验处理_ex.xls")
    df_asy = pd.read_excel("C:/Users/97899/Desktop/N/Network/spa_asy.xls")
    # 计算复杂度的波动程度
    var_comp = []
    asy=[]
    for i in range(38):
        l = []
        for j in range(1, 14):
            if df_comp.iloc[i, j] != -0.15:
                l.append(df_comp.iloc[i, j])
            else:
                l.append(-0.1)
        # print(std(l), (mean(l)))
        # if mean(l)!=0:
        var_comp.append(std(l)/mean(l))

    # plt.scatter(var_comp,df_asy.iloc[:,1].values)
    df=pd.DataFrame([var_comp,df_asy.iloc[:, 1].values,df_ex["氮素"].values,
                     df_ex["频率"].values,df_ex["刈割"].values]).T
    df.columns=["comp","asy", "N","F","M"]
    gb=df.groupby(["N"])
    comp=[]
    asy=[]
    for g in gb:
        gm=g[1].groupby(["F"])
        for m in gm:
            if float(g[0])>0:
                comp.append(log10(float(g[0]))+0.1)  #mean(m[1]["comp"])
                asy.append(mean(m[1]["asy"]))
            else:
                comp.append(0)
                asy.append(mean(m[1]["asy"]))

    print(comp,asy)
    y = asy
    x1 = comp
    # x2 = [[i, i ** 2] for i in x1]
    x = sm.add_constant(x1)
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    plt.scatter(x1, y,color='', marker='o', edgecolors='blue', s=100)
    y_fitted = results.fittedvalues
    x_y = sorted([(x, y) for x, y in zip(x1, y_fitted)])
    x_set = sorted(list(set(x_y)))
    x_mean = []
    y_mean = []
    for i in x_set:
        x_mean.append(i[0])
        y_ = []
        for jtem in x_y:
            if jtem[0] == i[0]:
                y_.append(jtem[1])
        y_mean.append(mean(y_))
    y_mean = array(y_mean)
    x_mean = array(x_mean)
    x_new = linspace(x_mean.min(), x_mean.max(), 300)
    y_smooth = make_interp_spline(x_mean, y_mean)(x_new)
    plt.plot(x_new, y_smooth, "black")
    # 复杂度的均值
    plt.xlabel("Log10 N addition rate"r"$(gNm^{-2}year^{-1})$", fontdict={"size": 12})#CV Complexity
    plt.ylabel("Species asynchrony")
    plt.text(0.25,0.72,r"$r^2=0.268*$")
    # r"$r^2=0.093,P=0.0628$"

    plt.show()
