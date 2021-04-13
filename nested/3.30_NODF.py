# -*- coding: utf-8 -*-
# @Time:2021/3/3014:16
# @File:3.30_null_model测试.py
import numpy as np
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


rng = np.random.RandomState(1)


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""构建随机矩阵"""


def null_mode(C_null):
    C_sim = C_null
    n = np.shape(C_sim)[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                C_sim[i, i] = 1
            else:
                s = rng.random()
                if s < C_sim[i, j]:
                    C_sim[i, j] = 1
                else:
                    C_sim[i, j] = 0

    return C_sim


"""确定度的顺序"""


def d_(C_, s="row"):
    n = np.shape(C_)[0]
    d_j = []
    if s != "row":
        C_ = C_.T
    for item in range(n):
        count = 0
        for j in range(n):
            if float(C_[item,j]) > 0.5:
                count += 1
        d_j.append(count)
    return d_j


"""计算nested"""

"""NODF只考虑度不一样的节点贡献"""


def nested(C_, d_1, d_2):
    n = np.shape(C_)[0]
    sum_1 = 0
    sum_2 = 0
    for i in range(n):
        for j in range(n):
            m1 = 0
            m2 = 0
            for l in range(n):
                # 列
                m1 = m1 + C_[i, l] * C_[j, l]  # 列不变，行变
                m2 = m2 + C_[l, i] * C_[l, j]  # 行不变，列变
            if d_1[i] > d_1[j]:  # 顺序正确，阶跃函数
                sum_1 += m1 / d_1[j]
            if d_2[i] > d_2[j]:
                sum_2 += m2 / d_2[j]
    return sum_1 + sum_2


"""计算NODF"""


def NPDF(C_):
    n = np.shape(C_)[0]
    """确定度(iteration)的顺序"""
    d_j = d_(C_)
    d_l = d_(C_, "col")
    S_NODF = nested(C_, d_j, d_l) / (n * (n - 1))
    return S_NODF


if __name__ == "__main__":
    """导入数据"""
    path = "C:/Users/97899/Desktop/N/"
    df_all_loop = pd.read_excel(path + "Network/all_loop_ex.xls")
    nest=[]
    nest_Z=[]
    for item in range(np.shape(df_all_loop)[0]):
        df_ex = df_all_loop.iloc[item, :]
        year = int(df_ex["year"])
        ex = df_ex["ex"]
        print(year,ex)
        dic = LoadDict(path + "N_pure/N_" + str(year) + "/Cmat.txt")
        """获得矩阵"""
        C_or = dic[ex]
        C_n = C_or.copy()
        C_simulate = null_mode(C_n)
        N_ = NPDF(C_simulate)
        nest.append(N_)
        """产生1000个null矩阵,算Z分数"""
        # sim = []
        # for item in range(1):
        #     # print("第"+str(item)+"次",C_or)
        #     C_origin = C_or.copy()  # 防止修改原地址的数值
        #     # 模拟产生的0-1矩阵
        #     C_simulate = null_mode(C_origin)
        #     sim.append(NPDF(C_simulate))
        # # print(sim)
        # # print(np.mean(sim))
        # # print(np.std(sim))
        # # print(N_)
        # Z=((N_ - np.mean(sim)) / np.std(sim))
        # # print("Z", Z)
        # nest_Z.append(nest_Z)
        # # plt.hist(sim, 30, density=1)
        # # plt.xlabel("NODF")
        # # # sns.distplot(sim,bins=30,hist=True,
        # # #              kde=True,kde_kws = {'color':'red', 'linestyle':'--'})
        # # plt.show()
        # # # if year==2008 or ex==10:
        # # #     break

    print(nest)
    df_all_loop["NODF"]=nest
    # df_all_loop["Z_NODF"] = nest_Z
    plt.hist(nest)
    plt.xlabel("NODF")
    plt.show()
    # df_all_loop.to_excel(path+"Network/loop_NODF.xls")