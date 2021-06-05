# -*- coding: utf-8 -*-
# @Time:2021/3/2921:10
# @File:3.29.py
import numpy as np
import random
import networkx as nx

random.seed(23)

nn=5
C = np.zeros(shape=(5, 5))
for i in range(5):
    for j in range(i, 5):
        if i == j:
            C[i, i] = 1
        else:
            r = random.random()
            C[i, j]=r
            C[j,i]=1-r
            # if r>0.6:
            #     C[i, j] = 1
            #     C[j, i] = 0
            # if r<=0.5:
            #     C[i, j] = 0
            #     C[j, i] = 1
print("原始矩阵",C)
A_row_mean = [np.sum(C[item, :]) for item in range(5)]
B_col_mean = [np.sum(C[:, item]) for item in range(5)]
C_row = np.column_stack((C, A_row_mean))
C_row=C_row[np.lexsort(-C_row.T)]
C_row = np.delete(C_row, 5, axis=1)
C_col = np.row_stack((C_row, B_col_mean))
C_col = C_col.T[np.lexsort(-C_col)].T
C = np.delete(C_col, 5, axis=0)
print("已排好序的矩阵",C )
"""确定度的顺序"""
d_j=[]
for item in range(5):
    count=0
    for j in C[item,:]:
        if j>0.5:
            count+=1
    d_j.append(count)
print("d_j",d_j)

# 计算nested
sum=0
for i in range(5):
    for j in range(i+1,5):
        # 两个循环完成i,j的行数
        m = 0
        for l in range(5):
            # 列
            m=m+C[i,l]*C[j,l]
        sum=sum+m/d_j[j]
k=(nn*(nn-1))
S_NODF=1/k*sum
print("结果1",S_NODF)


d_j=[]
for item in range(5):
    count=0
    for j in C[:,item]:
        if j>0.5:
            count+=1
    d_j.append(count)
print("d_j",d_j)
sum=0
for i in range(5):
    for j in range(i+1,5):
        # 两个循环完成i,j的行数
        m = 0
        for l in range(5):
            # 列
            m=m+C[l,i]*C[l,j]
        sum=sum+m/d_j[j]
k=(nn*(nn-1))
S_NODF1=1/k*sum
print("结果2",S_NODF1)
print("最终结果",S_NODF+S_NODF1)

def get_edge(C_mat, node):
    edge = []
    M = C_mat.shape[0]
    for i in range(M):
        for j in range(M):
            if C_mat[i, j] >= 0.8:
                edge.append((node[i], node[j]))
    return edge

node_list=[1,2,3,4,5]
G = nx.DiGraph()
G.add_nodes_from(node_list)  # 添加点a
edge_list = get_edge(C, node_list)
G.add_edges_from(edge_list) # 添加边
print(nx.is_bipartite(G))