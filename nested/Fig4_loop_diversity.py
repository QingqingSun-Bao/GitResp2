# -*- coding: utf-8 -*-
# @Time:2021/4/611:09
# @File:Fig4_loop_diversity.py
"""考察内嵌以及断环所在的网络在物种多样性上的差异"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from scipy import stats

if __name__=="__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_loop = pd.read_excel(path+"Network/loop_type.xls")
    df_rich = pd.read_excel(path+"Richness/rich_null21.xls",sheet_name="alpha")

    df_rich.set_index(["Unnamed: 0"],inplace=True)
    s_rich=[]
    n_rich=[]
    gb=df_loop.groupby(["year"])
    for g in gb:
        short_rich = []
        nest_rich = []
        for item in range(shape(g[1])[0]):
            df_ex = g[1].iloc[item, :]
            year = int(df_ex["year"])
            ex = df_ex["ex"]
            if int(df_ex["Short"]) ==1:
                short_rich.append(df_rich.loc[ex-1,year])
            elif int(df_ex["Nested"]) ==1:
                nest_rich.append(df_rich.loc[ex-1,year])
        if len(short_rich)==0:
            # 取最小值alpha6.6857,gamma=10.5,3
            s_rich.append(6.685714285714285)
        else:
            s_rich.append(mean(short_rich))
        if len(nest_rich) == 0:
            # 取最小值
            n_rich.append(7.4)
        else:
            n_rich.append(mean(nest_rich))
    print(s_rich,n_rich)
    diff=array(s_rich)-array(n_rich)
    print("检验结果",stats.ttest_1samp(diff,0))
    # 独立样本T检验（检验双样本均值）
    # Ttest_1sampResult(statistic=-5.568224361265347, pvalue=0.00012216779549730563)
    # plt.scatter(s_rich,n_rich)
    plt.plot(linspace(2008,2020,13),s_rich,"b",label="Short")
    plt.plot(linspace(2008, 2020, 13),n_rich,"r",label="Nested")
    plt.text(2014,11,"T-test=-5.568***")

    plt.xlabel("Year")
    plt.ylabel("Alpha Diversity")
    plt.legend()
    plt.show()


