# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from numpy import *
from itertools import combinations
from matplotlib.patches import ArrowStyle

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


def get_all_edge(C_mat, node):
    edge = []
    M = C_mat.shape[0]
    for i in range(M):
        for j in range(M):
            if C_mat[i, j] >= 0.5:
                edge.append((node[i], node[j]))
    return edge


def get_cyc_edge(C_mat, node, Ass):
    edge = []
    inx = []
    # print(node)
    for index, ass in enumerate(Ass):
        for no1 in node:
            if ass == no1:
                inx.append(index)
    two_node = list(combinations(inx, 2))
    # print(inx)
    # print(two_node)
    # print(len(Ass))
    for t_no in two_node:
        if C_mat[t_no[0], t_no[1]] > 0.5:
            # print(t_no[0], t_no[1])
            # print(Ass[t_no[0]])
            # print()
            edge.append((Ass[t_no[0]], Ass[t_no[1]]))
        else:
            edge.append((Ass[t_no[1]], Ass[t_no[0]]))
    return edge


def main():
    path = 'C:/Users/97899/Desktop/N/N_year/'
    # 各物种环的数量
    for year in range(2008, 2009):
        D = {}
        path1 = path + "N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        path3 = path + "N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        for ex in range(38, 39):#38,39
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
                node_list = Ass=['Stipa\ngrandis','Cleistogenes\nsquarrosa','Agropyron\ncristatum',
                                 'Allium\ntenuissimum','Koeleria\ncristata','Carex\nkorshinskyi']
                # '大针茅'Stipa grandis, '糙隐子草'Cleistogenes squarrosa, '冰草'Agropyron cristatum,
                # '洽草'Koeleria cristata, '细叶韭'Allium tenuissimum, '黄囊苔草'Carex korshinskyi
                print("Ass",node_list)
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.sans-serif'] = ['Times New Roman']
                G = nx.DiGraph()
                G.add_nodes_from(node_list)  # 添加点a
                edge_list = get_all_edge(C_mat, node_list)
                # print(edge_list)
                G.add_edges_from(edge_list)  # 添加边,起点为x，终点为y
                cyc_sys = list(nx.simple_cycles(G))
                # print(cyc_sys)

                '''显示图形'''
                # 结点分配不同的颜色
                pos=nx.circular_layout(G)
                #"yellow","violet","orange","cornflowerblue", "firebrick","lawngreen"
                node_clor=["turquoise"]
                nx.draw_networkx_nodes(G, pos,node_color=node_clor,
                                       with_labels=True, node_size=500)
                # 构建文本标签字典
                D_node = {}
                for ass in Ass:
                    D_node[ass] = ass
                # 添加结点标签
                nx.draw_networkx_labels(G, pos, labels=D_node, font_size=10)

                # 添加边
                ArrowStyle("wedge,tail_width=0.1,shrink_factor=0.8")
                # nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='r',
                #                        arrows=True,arrowsize=9,arrowstyle="wedge")
                # nx.draw_networkx_edges(G, pos=nx.circular_layout(G),edgelist=edge_list,
                #                        edge_color='r',arrows=True)


                # nx.draw(G, pos=nx.circular_layout(G), node_color=["limegreen", "r", "violet", "cyan", "orange", "yellow"],
                #         edge_color='red', with_labels=True,edgelist=[('糙隐子草', '糙隐子草'), ('星毛委陵菜', '糙隐子草')],
                #          font_size=10, node_size=3000)
                edge_ring=[]
                for item in cyc_sys:
                    if len(item)>3:
                        # if set(item)-set(['细叶韭', '洽草', '黄囊苔草', '糙隐子草'])==set():
                            # ['黄囊苔草', '糙隐子草', '洽草'],['细叶韭', '黄囊苔草', '糙隐子草']
                        # edge_list1 = get_cyc_edge(C_mat,item,Ass)
                        edge_list1=[('Allium\ntenuissimum', 'Koeleria\ncristata'), ('Koeleria\ncristata', 'Carex\nkorshinskyi'),
                                    ('Carex\nkorshinskyi', 'Cleistogenes\nsquarrosa'),('Cleistogenes\nsquarrosa', 'Allium\ntenuissimum')]
                        # '大针茅'Stipa grandis, '糙隐子草'Cleistogenes squarrosa, '冰草'Agropyron cristatum,
                        # '洽草'Koeleria cristata, '细叶韭'Allium tenuissimum, '黄囊苔草'Carex korshinskyi
                        nx.draw(G, pos,node_color=node_clor,
                                edge_color='red', with_labels=True, edgelist=edge_list1,width=1.5,
                                font_size=10, node_size=3000)
                        edge_ring.extend(edge_list1)
                print(edge_ring)
                edge_chain=set(edge_list)-set(edge_ring)-set([('Koeleria\ncristata', 'Allium\ntenuissimum')])
                nx.draw(G, pos,node_color=node_clor,
                                edge_color='darkcyan', with_labels=True, edgelist=edge_chain,width=0.7,
                                font_size=10, node_size=3000)
    plt.show()


main()
