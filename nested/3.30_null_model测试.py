# -*- coding: utf-8 -*-
# @Time:2021/3/3014:16
# @File:3.30_null_model测试.py
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)

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

"""FF-null model"""
# def null_mode(C_null,d_j,d_l):
#     C_sim = C_null
#     n = np.shape(C_sim)[0]
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 C_sim[i, i] = 1
#             else:
#                 if
#                 s = rng.random()
#                 if s < 0.5:
#                     C_sim[i, j] = 1
#
#         if C_sim[i, j]
#
#     return C_sim


"""构建原始矩阵"""


def origin_mat(C_ori):
    n = np.shape(C_ori)[0]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                C_ori[i, i] = 1
            else:
                r = rng.random()
                C_ori[i, j] = r
                C_ori[j, i] = 1 - r
    return C_ori


"""确定度的顺序"""


def d_(C_, s="row"):
    n = np.shape(C_)[0]
    d_j = []
    if s != "row":
        C_ = C_.T
    # print(np.shape(C_))
    for item in range(n):
        count = 0
        for j in C_[item, :]:
            if float(j) > 0.5:
                count += 1
        d_j.append(count)
    return d_j


"""计算nested"""


# def nested(C_,d_1,d_2):
#     n = np.shape(C_)[0]
#     sum_1 = 0
#     sum_2 = 0
#     # print(d_1,d_2)
#     for i in range(n):
#         for j in range(i + 1, n):
#             # 两个循环完成i,j的行数
#             m1 = 0
#             m2 = 0
#             for l in range(n):
#                 # 列
#                 m1 = m1 + C[i, l] * C[j, l]
#                 m2 = m2 + C[l, i] * C[l, j]
#             sum_1 = sum_1 + m1 / d_1[j]
#             sum_2 = sum_2 + m2 / d_2[j]
#     return sum_1+sum_2
"""NODF只考虑度不一样的节点贡献"""

def nested(C_, d_1, d_2):
    n = np.shape(C_)[0]
    sum_1 = 0
    sum_2 = 0
    # print(d_1,d_2)
    for i in range(n):
        for j in range(n):
            # 两个循环完成i,j的行数
            m1 = 0
            m2 = 0
            for l in range(n):
                # 列
                m1 = m1 + C[i, l] * C[j, l]  # 列不变，行变
                m2 = m2 + C[l, i] * C[l, j]  # 行不变，列变
            if d_1[i] > d_1[j]:  # 顺序正确，阶跃函数
                sum_1 += m1 / d_1[j]
            if d_2[i] > d_2[j]:
                sum_2 += m2 / d_2[j]
    return sum_1 + sum_2


"""计算NODF"""


def NPDF(C_):
    n = np.shape(C_)[0]
    # A_row_mean = [np.sum(C_[item, :]) for item in range(n)]
    # B_col_mean = [np.sum(C_[:, item]) for item in range(n)]
    # C_row = np.column_stack((C_, A_row_mean))
    # C_row = C_row[np.lexsort(-C_row.T)]
    # print(C_row)
    # C_row = np.delete(C_row, n, axis=1)
    # C_col = np.row_stack((C_row, B_col_mean))
    # C_col = C_col.T[np.lexsort(-C_col)].T
    # print("C_col",C_col)
    # C = np.delete(C_col, n, axis=0)
    # # print("已排好序的矩阵", C)
    """确定度(iteration)的顺序"""
    d_j = d_(C_)
    d_l = d_(C_, "col")
    S_NODF = nested(C_, d_j, d_l) / (n *(n-1))
    return S_NODF


if __name__ == "__main__":
    "产生1个原始矩阵"
    n = 5
    C = np.zeros(shape=(n, n))
    # print("原始矩阵",C_origin)
    """产生10000个null矩阵"""
    sim = []
    C_or = origin_mat(C)
    C_n = C_or.copy()
    C_simulate = null_mode(C_n)
    N_ = NPDF(C_simulate)

    for item in range(1000):
        # print("第"+str(item)+"次",C_or)
        C_origin = C_or.copy()  # 防止修改原地址的数值
        # 模拟产生的0-1矩阵
        C_simulate = null_mode(C_origin)
        print(C_simulate)
        sim.append(NPDF(C_simulate))
    print(sim)
    print(np.mean(sim))
    print(np.std(sim))
    print(N_)
    print("Z", (N_ - np.mean(sim) / np.std(sim)))
    plt.hist(sim)
    plt.show()
