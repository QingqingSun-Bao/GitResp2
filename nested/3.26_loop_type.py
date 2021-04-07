# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from numpy import *
# from Load_Save import LoadDict
from Load_Save import Savedict

"""Indenpend-loop的网络"""
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

'''导入C矩阵'''


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


'''筛选Spearman值大于0.6的物种组合'''


def Select_Zuhe(CP_dic, Ass_lst, Sp_lst):
    Ass = []
    max_spear = 0
    max_index = 0
    # print(Sp_lst)
    for item in range(len(Sp_lst)):
        if Sp_lst[item][0] > max_spear:
            max_spear = Sp_lst[item][0]
            max_index = item
    if max_spear >= 0.6:
        C_mat = CP_dic[max_index][0]
        Ass = Ass_lst[max_index]
    else:
        C_mat = np.zeros((3, 3))
    return C_mat, Ass


'''取出所有内嵌的三物种环'''


# def selec_three_circle(cir_lst, ass_lst):
#     D = []
#     for item in cir_lst:
#         if len[item] == 3:
#             A = [ass_lst[int(i)] for i in item]
#             # print(A)
#             D.append(A)
#     return D


'''取出所有长度大于1的loop'''


def sta_sys_three(sys_lst):
    D = {}
    th_ = []
    loop = []
    # print("打印整个列表",sys_lst)
    for item in sys_lst:
        if len(item) > 1:
            if len(item) in D.keys():
                D[len(item)] = D[len(item)] + 1
                if len(item) == 3:
                    th_.append(item)
                    loop.append(item)
            else:
                if len(item) == 3:
                    th_.append(item)
                D[len(item)] = 1
                loop.append(item)
    return D, loop


"""考察loop的独立性"""


def loog_short_loop(loop_lst, node_lst):
    # loop两两比较，是否存在相同的元素
    # print("导入的loop",loop_lst)
    node_length = len(node_lst)
    length = len(loop_lst)
    type_loop = []
    if length == 1:
        # print("short")
        type_loop.append(1)
    elif length > 1:
        for item in range(length - 1):
            for jtem in range(item+1, length):
                # print("两个物种组合",[set(loop_lst[item])&set(loop_lst[jtem])])
                A = set(loop_lst[item])
                B = set(loop_lst[jtem])
                if not bool(A & B):
                    # print("Independent",A,B)
                    type_loop.append(3)
                elif A.issubset(B) or B.issubset(A):
                    # A是B的子集，B是A的子集
                    print("nested",A,B)
                    type_loop.append(4)
                else:
                    # 不存在子集关系，只存在某几个交叉元素
                    # 环中有交叉
                    print("cross",A,B)
                    type_loop.append(5)
        for longem in loop_lst:
            if len(longem) == node_length:
                # print("long", node_lst)
                type_loop.append(2)
    return type_loop


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


# '''保存文件'''
#
#
# def savedict(datadict):
#     file = open("C:/Users/97899/Desktop/N/Zuhe/all_th_nest22.txt", "w")
#     file.write(str(datadict))
#     file.close()


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
    df_loop_ex = pd.read_excel(path + "Network/all_loop_ex.xls")
    lst_loop_type=[]
    columns=["year","ex","Short","Long","Independent","Nested","Cross","node_number"]
    for item in range(df_loop_ex.shape[0]):
        lst_type=[]
        year = df_loop_ex.iloc[item, 1]
        ex = float(df_loop_ex.iloc[item, 2])
        lst_type.extend([year,ex])
        # 获取物种组合
        path1 = path + "N_year/N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        # 获取Spearman值
        path3 = path + "N_year/N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        # 根据Spearman值筛选物种组合
        path2 = path + "N_year/N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
        CP_mat = LoadDict(path2)
        if year < 2016:
            C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
        else:
            C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])

        '''构建有向图'''
        node_list = Ass
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']
        G = nx.DiGraph()
        G.add_nodes_from(node_list)  # 添加点a
        edge_list = get_edge(C_mat, node_list)
        G.add_edges_from(edge_list)  # 添加边,起点为x，终点为y
        '''统计网络中环的类型：short1,long2,independent3,nested4,cross5'''
        # 统计环数：simple_cycles
        cyc_sys = list(nx.simple_cycles(G))
        cyc_dic, loop = sta_sys_three(cyc_sys)
        print(year, ex)
        type_loop = loog_short_loop(loop, Ass)
        # 全部化为0-1类型
        one_zero=[1 if i in set(type_loop) else 0 for i in range(1,6)]
        print(one_zero)
        lst_type.extend(one_zero)
        lst_type.extend([len(Ass)])
        lst_loop_type.append(lst_type)
        # print()
    pd.DataFrame(lst_loop_type,columns=columns).to_excel(path+"/Network/loop_cross.xls")









main()
