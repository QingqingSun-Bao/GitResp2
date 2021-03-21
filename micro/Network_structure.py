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
    return D


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
    path = 'C:/Users/97899/Desktop/N/N_year/'
    # 各物种环的数量
    write = pd.ExcelWriter("C:/Users/97899/Desktop/N/Network/circle21.xls")
    for year in range(2018, 2019):
        D = {}
        path1 = path + "N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        path3 = path + "N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        for ex in range(33, 34):
            D[ex] = {}
            ex = float(ex)
            path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
            CP_mat = LoadDict(path2)
            if year < 2016:
                C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
            else:
                C_mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
            if np.all(C_mat == 0):
                D[ex] = {3: -0.15}
            else:
                node_list = Ass
                # print(node_list)
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.sans-serif'] = ['SimHei']
                G = nx.DiGraph()
                G.add_nodes_from(node_list)  # 添加点a
                edge_list = get_edge(C_mat, node_list)
                print(edge_list)
                G.add_edges_from(edge_list)  # 添加边,起点为x，终点为y
                cyc_sys = list(nx.simple_cycles(G))
                print(year,ex)
                print(cyc_sys)
                print(nx.flow_hierarchy(G))
                D[ex] = sta_sys_three(cyc_sys)
                if not bool(D[ex]):
                    D[ex] = {3: 0}
                '''显示图形'''
                nx.draw(G, pos=nx.circular_layout(G), node_color='lightgreen', edge_color='black', with_labels=True,
                        font_size=10, node_size=3000)
                plt.show()
    #     F = pd.DataFrame.from_dict(D, orient='index')
    #     F_s = F.sort_index(axis=1).sort_index(axis=0)
    #     # 对行列索引排名
    #     F_s.fillna(0, inplace=True)
    #     # 对列排名
    #     F_s.to_excel(write, sheet_name=str(year))
    #     # print(D)
    # write.close()
    # write.save()



main()
