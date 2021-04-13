# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from numpy import *

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

'''导入数据'''


def LoadDataSet(CPmat):
    M = CPmat[0].shape[0]
    C_mat = np.mat(CPmat[0])
    # print('CPmat', M)
    g_mat = np.mat(np.zeros((M, M)))
    Tra_D = {}
    for i in range(M):
        Tra_D[i] = []
        for j in range(M):
            if i == j:
                g_mat[i, j] = 0
            else:
                if C_mat[i, j] > 0.5:
                    Tra_D[i].append(j)
                    g_mat[i, j] = 1
                else:
                    g_mat[i, j] = 0

    return C_mat, g_mat, Tra_D


'''判断是否有环'''


def findcircle(G):
    node_set = set()
    G = np.array(G)
    r = len(G)
    have_in_zero = True
    while have_in_zero:
        have_in_zero = False
        for i in range(r):
            if i not in node_set and not any([row[i] for row in G]):
                node_set.add(i)  # 节点列表
                G[i] = [0] * r
                have_in_zero = True
                break
    # print(node_set)
    return False if len(node_set) == r else True


'''找出有向图中所有的环'''
from copy import deepcopy as dc

# 用集合去除重复路径
ans = set()


def dfs(graph, trace, start):
    trace = dc(trace)  # 深拷贝，对不同起点，走过的路径不同
    # 如果下一个点在trace中，则返回环
    if start in trace:
        index = trace.index(start)
        tmp = [str(i) for i in trace[index:]]
        ans.add(str(''.join(tmp)))
        # str(' '.join(tmp))
        # print(trace[index:])
        return

    trace.append(start)

    # 深度优先递归递归
    for i in graph[start]:
        dfs(graph, trace, i)


'''统计环数'''


def Stasitccircle(ans_set):
    D_count = {}
    bns_set = list(ans_set)
    cns_set = list(ans_set)
    for i in range(len(bns_set)):
        for j in range(i + 1, len(bns_set)):
            if len(bns_set[i]) == len(bns_set[j]):
                if set(bns_set[i]) == set(bns_set[j]):
                    if bns_set[j] in cns_set:
                        cns_set.remove(bns_set[j])

    '''统计'''
    # print(cns_set)
    for item in cns_set:
        l = len(item)
        if l in D_count.keys():
            D_count[l] = D_count[l] + 1
        else:
            D_count[l] = 1
    # print(D_count)
    return D_count, cns_set


'''所有组合去重处理'''


def allcircle(D):
    allist = []
    for key in D.keys():
        allist.extend(D[key][1])
    # print('allist',allist)
    Count = Stasitccircle(allist)
    return Count


'''导入矩阵字典'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def get_edge(C_mat, node):
    edge = []
    M = C_mat.shape[0]
    for i in range(M):
        for j in range(M):
            if C_mat[i, j] >= 0.5:
                edge.append((node[i], node[j]))

    return edge


def main():
    path = 'C:/Users/97899/Desktop/locust/'
    D = {}
    D1 = {}
    for year in range(2012, 2014):
        D[str(year)] = {}
        path1 = path + "LX_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        for ex in Specise_set.keys():
            print("第几个实验",ex)
            max_spear=0
            max_inx=0
            indx=0
            for item in Specise_set[ex]:
                print(item)
                if max_spear < item[1]:
                    max_spear = item[1]
                    max_inx = indx
                indx += 1
            print("最大的矩阵",max_inx)
            path2 = path + "LX_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
            D_mat = LoadDict(path2)
            C_mat, G_mat, Tra_D = LoadDataSet(D_mat[max_inx])
            # print(C_mat)
            # C矩阵，有向图矩阵
            '''判断是否有环'''
            # have_circle = findcircle(G_mat)
            # print(have_circle)
            '''寻找有向图中的环'''
            dfs(Tra_D, [], 0)
            '''统计网络中的环数'''
            # D[str(year)][ex][item] = Stasitccircle(ans)  # 返回的两个值以元组的形式存储
            # ans.clear()
            # 清空集合
            print(str(year) + '年', '第' + str(ex) + '个实验',
                  Specise_set[ex][max_inx])
            node_list = Specise_set[ex][max_inx][0]
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = ['SimHei']
            G = nx.DiGraph()
            G.add_nodes_from(node_list)  # 添加点a
            edge_list = get_edge(C_mat, node_list)
            G.add_edges_from(edge_list)  # 添加边,起点为x，终点为y
            '''显示图形'''
            nx.draw(G, pos=nx.circular_layout(G), node_color='lightgreen', edge_color='black', with_labels=True,
                    font_size=10, node_size=3000)
            # plt.title('第'+str(ex)+'个实验')
            plt.show()
            # 实验汇总去重
        #     Count = allcircle(D[str(year)][ex])
        #     D1[ex] = Count[0]
        # F = pd.DataFrame.from_dict(D1, orient='index')
        # F.to_excel(path + '环数.xls')

    # print(D)


main()
