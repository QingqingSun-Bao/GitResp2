# -*- coding: utf-8 -*-
# @Time:2021/4/2018:57
# @File:Fig9_comp_bio_err.py
"""复杂度与生物量和株丛数之间的关系"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline


if __name__=="__main__":
    path="C:/Users/97899/Desktop/N/"
    df_comp=pd.read_excel(path+"network/loop_NODF.xls")
    df_bio=pd.read_excel(path+"Biomass/ramets_ex21.xls")
    # df_ramet=pd.read_excel(path+"Biomass/ramets_ex21.xls")
    bio_lst=[]
    comp_lst=[]
    bio_ = []
    comp_= []
    for i in range(np.shape(df_comp)[0]):
        df_ex=df_comp.iloc[i,:]
        year=int(df_ex["year"])
        ex=df_ex["ex"]
        # if df_ex["complexity"]<1:
        bio_lst.append(df_bio.loc[ex-1,year])
        comp_lst.append(df_ex["complexity"])
        # if df_ex["complexity"]==1:
        #     bio_.append(df_bio.loc[ex - 1, year])
        #     comp_.append(df_ex["complexity"])
    """按复杂度分组"""
    com_bio = [(x, y) for x, y in zip(comp_lst, bio_lst)]
    comp_set = list(set(com_bio))
    comp_mean = []
    bio_mean = []
    bio_err=[]
    for i in comp_set:
        comp_mean.append(i[0])
        y_ = []
        for jtem in com_bio:
            if jtem[0] == i[0]:
                y_.append(jtem[1])
        bio_mean.append(np.mean(y_))
        bio_err.append(np.std(y_))
    x1=comp_mean
    y=bio_mean
    # x1=comp_lst
    # y=bio_lst
    x2 = [[i, i ** 2] for i in x1]
    x=sm.add_constant(x2)
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    y_fitted = results.fittedvalues
    x_y = [(x, y) for x, y in zip(x1, y_fitted)]
    x_set = sorted(list(set(x_y)))
    x_mean = []
    y_mean = []
    for i in x_set:
        x_mean.append(i[0])
        y_ = []
        for jtem in x_y:
            if jtem[0] == i[0]:
                y_.append(jtem[1])
        y_mean.append(np.mean(y_))
    y_mean = np.array(y_mean)
    x_mean = np.array(x_mean)
    print(x_mean)
    x_new = np.linspace(x_mean.min(), x_mean.max(), 300)
    y_smooth = make_interp_spline(x_mean, y_mean)(x_new)
    plt.plot(x_new, y_smooth, "r")
    plt.scatter(x1,y)
    plt.errorbar(x1,y, yerr=bio_err,capsize=4,fmt="none")
    # plt.scatter(comp_, bio_,color="black")
    plt.xlabel("Complexity")
    plt.ylabel("Ramets")
    plt.text(0.1,350,r"$r^2=0.171***$")
    # Average plot Biomass
    plt.show()
