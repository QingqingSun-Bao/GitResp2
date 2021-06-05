# -*- coding: utf-8 -*-
# @Time:2021/3/3115:46
# @File:3.31_pure_mat.py
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from numpy import *
from Load_Save import Savedict
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

'''导入数据'''


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
    return C_mat, Ass,max_spear



'''导入矩阵字典'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic



def main():
    path = 'C:/Users/97899/Desktop/N/'
    # 各物种环的数量
    for year in range(2008, 2021):
        D = {}
        F={}
        S={}
        path1 = path + "N_year/N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        path3 = path + "N_year/N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        for ex in range(1, 39):
            ex = float(ex)
            path2 = path + "N_year/N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
            CP_mat = LoadDict(path2)
            if year < 2016:
                C_mat, Ass,Spear = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
            else:
                C_mat, Ass,Spear = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
            if np.all(C_mat == 0):
                continue
            else:
                D[ex]=C_mat
                F[ex]=[Ass]
                S[ex]=Spear
        Savedict(path+"N_pure/N_"+str(year)+"/Cmat.txt",D)
        Savedict(path + "N_pure/N_" + str(year) + "/Ass.txt", F)
        Savedict(path + "N_pure/N_" + str(year) + "/Spear.txt", S)







main()

