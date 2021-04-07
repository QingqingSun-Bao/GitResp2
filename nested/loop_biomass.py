# -*- coding: utf-8 -*-
# @Time:2021/4/613:33
# @File:loop_biomass.py
# -*- coding: utf-8 -*-
# @Time:2021/4/611:09
# @File:Fig4_loop_diversity.py
"""考察内嵌以及断环所在的网络在物种多样性上的差异"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_loop = pd.read_excel(path + "Network/loop_type.xls")
    df_biomass = pd.read_excel(path + "Biomass/bio_ex21.xls")

    df_biomass.set_index(["Unnamed: 0"], inplace=True)
    s_biomass = []
    n_biomass = []
    gb = df_loop.groupby(["year"])
    for g in gb:
        short_biomass = []
        nest_biomass = []
        for item in range(shape(g[1])[0]):
            df_ex = g[1].iloc[item, :]
            year = int(df_ex["year"])
            ex = df_ex["ex"]
            if int(df_ex["Short"]) == 1:
                short_biomass.append(df_biomass.loc[ex - 1, year])
            elif int(df_ex["Nested"]) == 1:
                nest_biomass.append(df_biomass.loc[ex - 1, year])
        if len(short_biomass) == 0:
            # 取最小值alpha6.6857,gamma=10.5,3
            s_biomass.append(104.97353333333334)
        else:
            s_biomass.append(mean(short_biomass))
        if len(nest_biomass) == 0:
            # 取最小值
            n_biomass.append(7.4)
        else:
            n_biomass.append(mean(nest_biomass))
    print(s_biomass, n_biomass)
    diff = array(s_biomass) - array(n_biomass)
    print("检验结果", stats.ttest_1samp(diff, 0))
    plt.plot(linspace(2008, 2020, 13), s_biomass, "b", label="Short")
    plt.plot(linspace(2008, 2020, 13), n_biomass, "r", label="Nested")
    plt.text(2014, 300, "T-test=2.416*")

    plt.xlabel("Year")
    plt.ylabel("Biomass in a plot")
    plt.legend()
    plt.show()
