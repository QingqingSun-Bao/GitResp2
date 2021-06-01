# -*- coding: utf-8 -*-
# @Time:2021/4/613:33
# @File:Fig4_loop_biomass.py
# -*- coding: utf-8 -*-
# @Time:2021/4/611:09
# @File:Fig4_loop_diversity.py
"""考察内嵌以及断环所在的网络在物种生物量以及株丛数上的差异"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_loop = pd.read_excel(path + "Network/loop_type.xls")
    df_biomass = pd.read_excel(path + "Biomass/ramets_ex21.xls")

    # df_biomass.set_index(["Unnamed: 0"], inplace=True)
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


    diff = array(s_biomass) - array(n_biomass)
    print("检验结果", stats.ttest_1samp(diff, 0))

    fig,ax1=plt.subplots()
    x=linspace(1, 13, 13)
    width=0.35
    plt.bar(x-width/2,s_biomass,width,label="Short")
    plt.bar(x + width / 2, n_biomass, width, label="Nest")
    plt.axis("tight")
    plt.xlabel("Year")
    plt.ylabel("Ramets")
    plt.text(3, 300, "T-test=-0.784")
    labels = [str(int(i)) for i in linspace(2008, 2020, 13)]
    plt.xticks(x, labels, rotation=30)


    # 副坐标轴
    ax2=ax1.twinx()
    cha=[ abs(x1-y1) for x1,y1 in zip(s_biomass,n_biomass)]
    print("cha",cha)
    plt.plot(x, cha, "black")
    plt.ylabel("Difference value")
    # plt.plot(linspace(2008, 2020, 13), n_biomass, "r", label="Nested")

    # 添加到内部
    # x1=linspace(1, 6, 13)
    # ax1.set(aspect=1, xlim=(1, 6), ylim=(350, 500))
    # axins = zoomed_inset_axes(ax1, zoom=2, loc='upper left')
    # im = axins.imshow(x, extent=cha, origin="lower")
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)

    # Biomass in a plot
    # plt.legend()
    plt.show()
