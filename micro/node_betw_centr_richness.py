# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from numpy import *
from networkx.algorithms import tournament

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

'''导入数据'''

def LoadDataSet(C_mat):
    M = C_mat.shape[0]
    C_mat = np.mat(C_mat)
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

    return g_mat, Tra_D

def Select_Zuhe(CP_dic, Ass_lst, Sp_lst):
    # print(Ass_lst)
    Ass = []
    max_spear = 0
    max_index = 0
    for item in range(len(Sp_lst)):
        # print(item, len(Sp_lst))
        if Sp_lst[item][0] > max_spear:
            # print(max_spear,i)
            max_spear = Sp_lst[item][0]
            max_index = item
    if max_spear >= 0.6:
        # print(CP_dic)
        C_mat = CP_dic[max_index][0]
        P_mat = CP_dic[max_index][1]
        Ass = Ass_lst[max_index]
    else:
        C_mat = np.zeros((3, 3))
        P_mat = np.zeros((3, 3))
        max_spear = -0.15
    # print(C_mat,max_spear)
    return C_mat, Ass


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
    allist.extend(D[1])
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
    path = 'C:/Users/97899/Desktop/N/'
    single_nested=LoadDict(path+"Zuhe/single_nested_ex.txt")
    rich=pd.read_excel(path+"Richness/rich_null.xls")
    strong=pd.read_excel(path+"Network/Strong_index.xls")
    print(strong)
    A=[]
    for item in single_nested:
        for i in single_nested[item]:
            if i in A:
                continue
            else:
                A.append(i)
    # print(A,len(A))
    # 各物种环的数量
    D={}
    D_rich={}
    D_rich["cent"],D_rich["rich"]=[],[]
    for year in range(2009, 2021):
        path1 = path + "N_year/N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        path3 = path + "N_year/N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        for ex in range(1, 39):
            ex = float(ex)
            if [year,ex] not in A:
                path2 = path + "N_year/N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
                CP_mat = LoadDict(path2)
                if year < 2016:
                    C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
                else:
                    C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
                if np.all(C_mat == 0):
                    continue
                else:
                    node_list = Ass
                    G = nx.DiGraph()
                    G.add_nodes_from(node_list)  # 添加点a
                    edge_list = get_edge(C_mat, node_list)
                    G.add_edges_from(edge_list)  # 添加边,起点为x，终点为y
                    cent=nx.betweenness_centrality(G)
                    # print(Ass)
                    print(nx.betweenness_centrality(G))
                    max_centr=max(cent.values())
                    print(max(cent.values()))
                    '''取字典中的最大值'''
                    # 考察最大中介中心性与物种多样性的关系
                    if max_centr>0 and strong.loc[ex-1,year]>-0.15:
                        D_rich["cent"].append(max_centr)
                        D_rich["rich"].append(strong.loc[ex-1,year])
                    for jtem in cent:
                        if jtem not in D.keys():
                            D[jtem]=[cent[jtem],1]
                        else:
                            D[jtem]=[D[jtem][0]+cent[jtem],D[jtem][1]+1]
                    '''显示图形'''
                    plt.rcParams['axes.unicode_minus'] = False
                    plt.rcParams['font.sans-serif'] = 'SimHei'
                    nx.draw(G, pos=nx.circular_layout(G), node_color='lightgreen', edge_color='black', with_labels=True,
                            font_size=10, node_size=3000)
                    # plt.show()
    # for item in D.keys():
    #     print(item,D[item][0]/D[item][1])
    print(D_rich)
    plt.scatter(D_rich["rich"],D_rich["cent"])
    plt.show()

main()
