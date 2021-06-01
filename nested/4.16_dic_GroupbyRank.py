# -*- coding: utf-8 -*-
# @Time:2021/4/1614:15
# @File:4.16_dic_GroupbyRank.py
import pandas as pd
from numpy import *
import numpy as np


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic

def Savedict(path,datadict):
    file = open(path, "w",encoding='utf8')
    file.write(str(datadict))
    file.close()


"""给物种分等级"""

def get_rank(lst):
    lst_rank=[]
    rank=[0.8,0.7,0.6,0.5,0.4,0.3,0.2]
    #0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0
    for l in lst:
        for i in range(len(rank)):
           if l>rank[i]:
               lst_rank.append(i+1)
               break
    return lst_rank


def get_groupbycompete(c_mat):
    row_mean=[]
    n=np.shape(c_mat)[0]
    for i in range(n):
        row_mean.append(np.mean(c_mat[i,:]))
    rank = get_rank(row_mean)
    return rank



if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_NODF = pd.read_excel(path + "Network/loop_NODF.xls")
    dic_zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    # 等级字典
    key=["1","2","3","4","5","6","7"]
    #"1","2","3","4","5","6","7","8","9","10","11","12"
    dic_com = dict([(k,[]) for k in key])
    print(dic_com)
    """将所有的物种分等级整理"""
    for item in range(np.shape(df_NODF)[0]):
        df_ex = df_NODF.iloc[item, :]
        year = int(df_ex["year"])
        ex = df_ex["ex"]
        print(year, ex)
        # 引入竞争矩阵
        dic_matic = LoadDict(path + "N_pure/N_" + str(year) + "/Cmat.txt")
        rank=get_groupbycompete(dic_matic[ex])
        for spec,r in zip(dic_zuhe[year][ex],rank):
            dic_com[str(r)].append([year, ex, spec])

    print(dic_com)
    for key1 in dic_com.keys():
        print(len(dic_com[key1]))
    Savedict(path+"Attribute/groupbyrank_0.8-0.txt",dic_com)
    # 47
    # 108
    # 82
    # 58
    # 73
    # 95
    # 141
    # 153
    # 162
    # 118
    # 79
    # 54


