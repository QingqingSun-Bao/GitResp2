# -*- coding: utf-8 -*-
# @Time:2021/4/911:16
# @File:4.9_degeneracyINgroup.py
#
import numpy as np
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from scipy import stats
import statsmodels.api as sm

"""分组查看complexity/NODF"""


def group_by(df, index, str1, str2):
    no_com_M = []
    no_com_NM = []
    gb = df.groupby([str1])
    for g in gb:
        g1 = g[1].groupby([str2])
        for g2 in g1:
            n = len(g2[1][index])
            if int(g2[0]) == 0:
                no_com_NM.append(np.mean(g2[1][index].tolist() + np.repeat(0, 36 - n).tolist()))

            else:
                no_com_M.append(np.mean(g2[1][index].tolist() + np.repeat(0, 36 - n).tolist()))

    return no_com_M, no_com_NM


"""检查NODF/Complexity与简并度的关系受氮素的影响"""


def NODF_degen_groupby(df, index, str1, str2):
    no_com_M = []
    no_com_NM = []
    gb = df.groupby([str1])
    for g in gb:
        g1 = g[1].groupby([str2])
        for g2 in g1:
            n = len(g2[1][index])
            if int(g2[0]) == 0:
                if n == 1:
                    no_com_NM.append(0)
                else:
                    no_com_NM.append(stats.pearsonr(g2[1][index], g2[1]["degeneracy"])[0])
            else:
                no_com_M.append(stats.pearsonr(g2[1][index], g2[1]["degeneracy"])[0])

    return no_com_M, no_com_NM


if __name__ == "__main__":
    """导入数据"""
    path = "C:/Users/97899/Desktop/N/"
    df_NODF_deg = pd.read_excel(path + "Network/NODF_deg.xls")
    print("复杂度与简并度", stats.pearsonr(df_NODF_deg["complexity"], df_NODF_deg["degeneracy"]))
    print("NODF与简并度", stats.pearsonr(df_NODF_deg["NODF"], df_NODF_deg["degeneracy"]))
    print("NODF与复杂度", stats.pearsonr(df_NODF_deg["NODF"], df_NODF_deg["complexity"]))
    x = np.log10([1, 2, 3, 4, 6, 11, 16, 21, 51])

    """检测N素与刈割下，nested的程度与简并度的关系变化"""

    # re_M, re_NM = NODF_degen_groupby(df_NODF_deg, "NODF", "N", "Mow")
    # plt.scatter(x, re_M, c="b", facecolor=False)
    # plt.scatter(x, re_NM, c="orange", facecolor=False)
    # sb.regplot(x, re_M, fit_reg=True, scatter_kws={"alpha": 1 / 3},
    #            color="blue", marker="o", label="Mowing")
    # sb.regplot(x, re_NM, fit_reg=True, scatter_kws={"alpha": 1 / 3},
    #            color="orange", marker="o", label="No-Mowing")


    """检测N素与刈割下，复杂度的差异"""
    r_M, r_NM = group_by(df_NODF_deg, "complexity", "N", "Mow")

    x1 = sm.add_constant(x)
    model = sm.OLS(r_M, x1)
    results = model.fit()
    print("刈割",results.summary())
    model = sm.OLS(r_NM, x1)
    results = model.fit()
    print("不刈割", results.summary())

    sb.regplot(x, r_M, fit_reg=True, scatter_kws={"alpha": 1 / 3},
               color="red", marker="o", label="Mowing")
    # degrneracy
    # plt.text(1.25, 0.3, r"$r^2=0.919***$")
    # plt.text(0.25, 0.05, r"$r^2=0.844***$")
    sb.regplot(x, r_NM, fit_reg=True, scatter_kws={"alpha": 1 / 3},
               color="blue", marker="o", label="No-Mowing")
    # complexity
    plt.text(1.25, 0.3, r"$r^2=0.785**$")
    plt.text(0.25, 0.05, r"$r^2=0.846***$")

    plt.xlabel("N")
    plt.ylabel("Complexity")
    plt.legend()
    plt.show()
    """画出多个子图"""
    # fig, axs = plt.subplots(2, 2)
    #
    # axs[0, 0].scatter(Node_num, df_NODF_deg["NODF"])
    # axs[0, 0].set_ylabel("NODF")
    # axs[0, 0].set_title("Node_num")
    # axs[0, 0].text(4,0.9,"0.281***")
    #
    # axs[0, 1].scatter(df_NODF_deg["degeneracy"], df_NODF_deg["NODF"])
    # axs[0, 1].set_title("Degeneracy")
    # axs[0, 1].text(0.9, 0.9, "-0.566***")
    #
    # axs[1, 0].scatter(Node_num, df_NODF_deg["complexity"])
    # axs[1, 0].set_ylabel("Complexity")
    # axs[1, 0].text(4, 0.8, "0.458***")
    #
    # axs[1, 1].scatter(df_NODF_deg["degeneracy"], df_NODF_deg["complexity"])
    # axs[1, 1].text(0.9, 0.8, "0.379***")
    #
    #
    #
    # plt.show()
