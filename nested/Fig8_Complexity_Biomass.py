# -*- coding: utf-8 -*-
# @Time:2021/4/1915:26
# @File:Fig8_Complexity_Biomass.py
"""NODF与生物量、株丛数之间的关系"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline

if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_NODF = pd.read_excel(path + "Network/loop_NODF.xls")
    df_bio = pd.read_excel(path + "Biomass/bio_ex21.xls")
    bio_lst = {}
    Comp = {}
    """所有的高低氮素"""
    # bio_lst["h"] = []
    # bio_lst["l"] = []
    # Comp["h"] = []
    # Comp["l"] = []
    # gb = df_NODF.groupby("N")
    # for g in gb:
    #     print(g)
    #     for item in range(np.shape(g[1])[0]):
    #         df_ex = g[1].iloc[item, :]
    #         # print(item,df_ex)
    #         year = int(df_ex["year"])
    #         ex = df_ex["ex"]
    #         bio = df_bio[year].values.tolist()
    #         if year < 2008:
    #             bio_lst["l"].append(bio[int(ex) - 1])
    #             Comp["l"].append(df_ex["complexity"])
    #         if year > 2008:
    #             if int(g[0]) >= 2:
    #                 bio_lst["h"].append(bio[int(ex) - 1])
    #                 Comp["h"].append(df_ex["complexity"])
    #             if int(g[0]) < 2:
    #                 bio_lst["l"].append(bio[int(ex) - 1])
    #                 Comp["l"].append(df_ex["complexity"])
    #
    # print(bio_lst)
    """所有的氮素"""
    # gb = df_NODF.groupby("Mow")
    # for g in gb:
    #     bio_lst[g[0]] = []
    #     Comp[g[0]] = []
    #     for item in range(np.shape(g[1])[0]):
    #         df_ex = g[1].iloc[item, :]
    #         year = int(df_ex["year"])
    #         ex = df_ex["ex"]
    #         bio = df_bio[year].values.tolist()
    #         bio_lst[g[0]].append(bio[int(ex) - 1])
    #         Comp[g[0]].append(df_ex["complexity"])
    # print(bio_lst)
    """频率"""
    gb = df_NODF.groupby("Fre")
    for g in gb:
        bio_lst[g[0]] = []
        Comp[g[0]] = []
        for item in range(np.shape(g[1])[0]):
            df_ex = g[1].iloc[item, :]
            year = int(df_ex["year"])
            ex = df_ex["ex"]
            bio = df_bio[year].values.tolist()
            # if year==2008:
            #     bio_lst[0].append(bio[int(ex) - 1])
            #     Comp[0].append(df_ex["complexity"])
            # else:
            #     bio_lst[g[0]].append(bio[int(ex) - 1])
            #     Comp[g[0]].append(df_ex["complexity"])
            bio_lst[g[0]].append(bio[int(ex) - 1])
            Comp[g[0]].append(df_ex["complexity"])
    print(bio_lst)
    # """取均值"""
    # bio_mean = {}
    # Comp_lst={}
    # for key in bio_lst.keys():
    #     gbc = set(Comp[key])
    #     bio_mean[key]=[]
    #     Comp_lst[key]=[]
    #     for i in gbc:
    #         bio_m=[]
    #         for index, c in enumerate(Comp[key]):
    #             if c == i:
    #                 bio_m.append(bio_lst[key][index])
    #         bio_mean[key].append(np.mean(bio_m))
    #         Comp_lst[key].append(i)
    # print(bio_mean)
    # print(Comp_lst)

    """画图并检验"""
    ex = 12
    y = bio_lst[0]+bio_lst[2]
    x1 = Comp[0]+Comp[2]
    # print(max(y),len(y),len(bio_lst["l"]))
    # y = bio_lst
    # x1 = df_NODF("complexity")
    x2 = [[i, i ** 2] for i in x1]
    x = sm.add_constant(x1)
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    plt.scatter(x1,y)
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
        y_mean.append(np.mean(y_))
    y_mean = np.array(y_mean)
    x_mean = np.array(x_mean)
    print(x_mean)
    x_new = np.linspace(x_mean.min(), x_mean.max(), 300)
    y_smooth = make_interp_spline(x_mean, y_mean)(x_new)
    plt.plot(x_new, y_smooth, "r")
    plt.xlabel("Complexity(Low Frequency)", fontdict={"size": 15})
    plt.ylabel("Average plot biomass", fontdict={"size": 15})
    plt.text(0.8, 350, r"$r^2=0.004$", fontdict={"size": 13})
    plt.show()
