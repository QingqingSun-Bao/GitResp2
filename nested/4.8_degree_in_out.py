# -*- coding: utf-8 -*-
# @Time:2021/4/820:18
# @File:4.8_degree_in_out.py
# -*- coding: utf-8 -*-
# @Time:2021/4/712:51
# @File:4.8_degeneracyInDegree.py
# -*- coding: utf-8 -*-
# @Time:2021/3/3014:16
# @File:3.30_null_model测试.py
import numpy as np
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import collections
from scipy import stats

rng = np.random.RandomState(1)


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""确定度的顺序"""


def d_(C_, s="row"):
    n = np.shape(C_)[0]
    d_j = []
    if s != "row":
        C_ = C_.T
    for item in range(n):
        count = 0
        for j in range(n):
            if float(C_[item, j]) > 0.5:
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


"""计算number of specise with the same degree"""


def same_degree(d_j):
    count = 0
    degree_dic = collections.Counter(d_j)
    for key in degree_dic.keys():
        if degree_dic[key] > 1:
            count += degree_dic[key]
    degeneracy = count / len(d_j)

    return degeneracy


"""计算link出度的相关系数"""


def degree_out_in(C_mat):
    n = np.shape(C_mat)[0]
    d_r = d_(C_mat, s="row")
    out_1 = []
    out_2 = []
    in_1=[]
    in_2=[]
    for i in range(n):
        for j in range(n):
            if i!=j :
                if C_mat[i, j] > 0.6:
                    out_1.append(d_r[i])
                    out_2.append(d_r[j])
                else:
                    in_1.append(n-d_r[i])
                    in_2.append(n-d_r[j])
    r_out,p_out=stats.pearsonr(out_1,out_2)
    r_in, p_in = stats.pearsonr(in_1, in_2)
    print(out_2,out_1)
    print("出度",r_out,p_out)
    print("入度", r_in, p_in)
    if r_in is NaN:
        r_in=0
    if r_out is NaN:
        r_out=0
    return r_out,r_in


"""计算NODF"""


def NPDF(C_):
    n = np.shape(C_)[0]
    """确定度(iteration)的顺序"""
    d_j = d_(C_)
    d_l = d_(C_, "col")
    deg = same_degree(d_j)
    S_NODF = nested(C_, d_j, d_l) / (n * (n - 1))
    return S_NODF, deg


if __name__ == "__main__":
    """导入数据"""
    path = "C:/Users/97899/Desktop/N/"
    df_all_loop = pd.read_excel(path + "Network/all_loop_ex.xls")
    nest = []
    nest_deg = []
    Node_num = []
    r_out_degree = []
    r_in_degree = []
    for item in range(np.shape(df_all_loop)[0]):
        df_ex = df_all_loop.iloc[item, :]
        year = int(df_ex["year"])
        ex = df_ex["ex"]
        print(year, ex)
        dic = LoadDict(path + "N_pure/N_" + str(year) + "/Cmat.txt")
        """获得矩阵"""
        C_or = dic[ex]
        C_n = C_or.copy()
        Node_num.append(np.shape(C_or)[0])
        N_, deg = NPDF(C_or)
        nest.append(N_)
        nest_deg.append(deg)
        de=degree_out_in(C_n)
        r_out_degree.append(de[0])
        r_in_degree.append(de[1])

    df_all_loop["NODF"] = nest
    df_all_loop["degeneracy"] = nest_deg
    # df_all_loop.to_excel(path + "Network/NODF_deg.xls")

    """检验相关性"""

    print("NODF与出度相关性", stats.pearsonr(df_all_loop["NODF"], r_out_degree))
    print("复杂度与出度相关性", stats.pearsonr(df_all_loop["complexity"], r_out_degree))
    print("NODF与入度相关性", stats.pearsonr(df_all_loop["NODF"], r_in_degree))
    print("复杂度与入度相关性", stats.pearsonr(df_all_loop["complexity"], r_in_degree))

    """画出多个子图"""
    fig,axs=plt.subplots(2,2)

    axs[0,0].scatter(r_in_degree,df_all_loop["NODF"])
    axs[0, 0].set_ylabel("NODF")
    axs[0,0].set_title("r-In_Degree")
    axs[0, 0].plot([0,0],[0.2,1],c="red")
    # plt.axvline()

    axs[0, 1].scatter(r_out_degree,df_all_loop["NODF"])
    axs[0, 1].set_title("r-Out_Degree")
    axs[0, 1].plot([0, 0], [0.2, 1], c="red")

    axs[1, 0].scatter(r_in_degree,df_all_loop["complexity"])
    axs[1, 0].set_ylabel("Complexity")
    axs[1, 0].plot([0, 0], [0, 1], c="red")

    axs[1, 1].scatter(r_out_degree,df_all_loop["complexity"])
    axs[1, 1].plot([0, 0], [0, 1], c="red")

    # # plt.scatter(r_in_degree, df_all_loop["complexity"])
    # plt.xlabel("r_in_disassortativity")
    # plt.ylabel("complexity")
    # # plt.text(0.8, 0.9, "R2=-0.566***")
    plt.show()

