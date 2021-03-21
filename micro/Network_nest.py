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


'''筛选物种组合'''


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


'''取出所有内嵌的三物种环'''


def selec_three_circle(cir_lst, ass_lst):
    D = []
    for item in cir_lst:
        if len(item) == 3:
            A = [ass_lst[int(i)] for i in item]
            # print(A)
            D.append(A)
    return D


def sta_sys_three(sys_lst):
    D = {}
    th_ = []
    for item in sys_lst:
        if len(item) > 1:
            if len(item) in D.keys():
                D[len(item)] = D[len(item)] + 1
                if len(item) == 3:
                    th_.append(item)
            else:
                if len(item) == 3:
                    th_.append(item)
                D[len(item)] = 1
    return D ,th_


'''统计三物种环出现的次数'''


def sta_fre_th(D_all, th_lst):
    D_lst = list(D_all.keys())
    for thtem in th_lst:
        # print(thtem)
        if tuple(thtem) not in D_lst:
            for Dtem in D_lst:
                # 再次判断集合是否存在
                if set(thtem) == set(Dtem):
                    D_all[tuple(Dtem)] = D_all[tuple(Dtem)] + 1
                else:
                    D_all[tuple(thtem)] = 1
        else:
            D_all[tuple(thtem)] = D_all[tuple(thtem)] + 1
    # print(D_all)
    return D_all


'''导入矩阵字典'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''保存文件'''


def savedict(datadict):
    file = open("C:/Users/97899/Desktop/N/Zuhe/all_th_nest22.txt", "w")
    file.write(str(datadict))
    file.close()


def get_edge(C_mat, node):
    edge = []
    M = C_mat.shape[0]
    for i in range(M):
        for j in range(M):
            if C_mat[i, j] >= 0.5:
                edge.append((node[i], node[j]))

    return edge


def main():
    path = 'C:/Users/97899/Desktop/N/N_year/'
    D = {}
    D1 = {}
    th_D = {}
    n = 0
    m = 0
    for year in range(2008, 2021):
        D[year] = {}
        D1[year] = {}
        th_D[year] = {}
        # 获取物种组合
        path1 = path + "N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        # 获取Spearman值
        path3 = path + "N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        # 根据Spearman值筛选物种组合
        for ex in range(1, 39):
            ex = float(ex)
            path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
            CP_mat = LoadDict(path2)
            if year < 2016:
                C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
            else:
                C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
            if np.all(C_mat == 0):
                D1[year][ex] = [-0.15]
                th_D[year][ex] = [[-0.15]]
                continue
            else:
                '''统计网络中的环数'''
                node_list = Ass
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.sans-serif'] = ['SimHei']
                G = nx.DiGraph()
                G.add_nodes_from(node_list)  # 添加点a
                edge_list = get_edge(C_mat, node_list)
                G.add_edges_from(edge_list)  # 添加边,起点为x，终点为y
                cyc_sys = list(nx.simple_cycles(G))
                cyc_dic,th_zuhe = sta_sys_three(cyc_sys)
                # print("sys", cyc_dic,th_zuhe )
                '''显示图形'''
                nx.draw(G, pos=nx.circular_layout(G), node_color='lightgreen', edge_color='black', with_labels=True,
                        font_size=10, node_size=3000)
                # plt.show()
                n = n + 1
                if 3 in cyc_dic.keys():
                    # 三个环的数量
                    D1[year][ex] = cyc_sys[3]
                    # 各三环对应的物种列表
                    th_D[year][ex] = th_zuhe
                    m = m + 1
                else:
                    th_D[year][ex] = [[0]]
    # F = pd.DataFrame.from_dict(D1, orient='index')
    # pd.DataFrame(D1).to_excel(path + 'Network/环数.xls')
    print("有三物种环的占比%.3f,共有%d个实验存在环，以竞争主导的实验样地有%d个" % (m / n, m, n))
    savedict(th_D)
    # print(sorted(th_order.items(), key=lambda x: x[1], reverse=True))


main()
