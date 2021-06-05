# -*- coding: utf-8 -*-
# @Time:2021/4/1510:25
# @File:4.15_max_competion.py
"""考察网络内竞争系数的影响"""
import pandas as pd
from numpy import *
import numpy as np


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def get_compete(C_mat):
    n = np.shape(C_mat)[0]
    mean_row = []
    for i in range(n):
        mean_row.append(np.mean(C_mat[i, :]))
    max_value = max(mean_row)
    min_value=min(mean_row)
    mean_value=mean(mean_row)
    return max_value, min_value,mean_value


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_NODF_compete = pd.read_excel(path + "Network/loop_NODF.xls")
    max_compete = []
    min_compete=[]
    mean_compete=[]
    for item in range(np.shape(df_NODF_compete)[0]):
        df_ex = df_NODF_compete.iloc[item, :]
        year = int(df_ex["year"])
        ex = df_ex["ex"]
        print(year, ex)
        dic = LoadDict(path + "N_pure/N_" + str(year) + "/Cmat.txt")
        """获得矩阵"""
        C_or = dic[ex]
        Max_value, Min_value,Mean_value = get_compete(C_or)
        max_compete.append(Max_value)  # 获取竞争系数最大值
        min_compete.append(Min_value)
        mean_compete.append(Mean_value)
    df_NODF_compete["min_compete"]=min_compete
    df_NODF_compete["max_compete"]=max_compete
    df_NODF_compete["mean_compete"]=mean_compete
    # df_NODF_compete.to_excel(path+"Network/NODF_compete-1.xls")
    gb=df_NODF_compete.groupby("N")
    for g in gb:
        print(g[0])
        print(mean(g[1].loc[:,"mean_compete"]))

