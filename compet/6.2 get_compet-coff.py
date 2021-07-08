# -*- coding: utf-8 -*-
# @Time:2021/6/216:51
# @File:6.2 get_compet-coff.py
import pandas as pd
from numpy import *
import numpy as np


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def Savedict(path, datadict):
    file = open(path, "w", encoding='utf8')
    file.write(str(datadict))
    file.close()


"""各网络物种的平均竞争系数"""


def get_groupbycompete(c_mat):
    row_mean = []
    n = np.shape(c_mat)[0]
    for i in range(n):
        row_mean.append(np.mean(c_mat[i, :]))
    return row_mean


"""计算各组平均系数落入区间的频数"""


def get_frequency(lst):
    rank = [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0]
    # 0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0
    lst_rank = [0 for i in range(len(rank))]
    for l in lst:
        for i in range(len(rank)):
            if l > rank[i]:
                lst_rank[i] += 1
                break
    return lst_rank


"""汇总38个网络，打印出表格"""

if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_Complexity = pd.read_excel(path + "Network/Strong_index.xls")
    dic_zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    # 各网络物种的平均竞争系数
    dic_fre = dict([(k + 1, []) for k in range(38)])
    dic_comp = dict([(k + 1, []) for k in range(38)])
    n, m = shape(df_Complexity)
    for c in range(1, m):
        year=c+2007
        for r in range(n):
            ex=r+1
            # 判断是否为链
            if df_Complexity.iloc[r,c]==0:
                dic_matic = LoadDict(path + "N_pure/N_" + str(year) + "/Cmat.txt")
                print("链")
                comp = get_groupbycompete(dic_matic[ex])
                dic_comp[ex].append(comp)
                # 计算各组平均系数落入区间的频数
                fre = get_frequency(comp)
                dic_fre[ex].append(fre)
    Savedict(path + "Competition/dic_comp_chain.txt", dic_comp)
    Savedict(path + "Competition/dic_fre_chain.txt", dic_fre)

    """非传递性网络中的竞争系数"""
    # df_NODF = pd.read_excel(path + "Network/loop_NODF.xls")
    # for item in range(np.shape(df_NODF)[0]):
    #     df_ex = df_NODF.iloc[item, :]
    #     year = int(df_ex["year"])
    #     ex = df_ex["ex"]
    #     print(year, ex)
    #     # 引入竞争矩阵
    #     dic_matic = LoadDict(path + "N_pure/N_" + str(year) + "/Cmat.txt")
    #     comp = get_groupbycompete(dic_matic[ex])
    #     dic_comp[ex].append(comp)
    #     # 计算各组平均系数落入区间的频数
    #     fre = get_frequency(comp)
    #     dic_fre[ex].append(fre)
    # Savedict(path + "Competition/dic_comp.txt", dic_comp)
    # Savedict(path + "Competition/dic_fre.txt", dic_fre)

    # 汇总38个网络，打印出表格
