# -*- coding: utf-8 -*-
# @Time:2021/6/169:56
# @File:6.16 Statistic.py
"""统计各氮素下非传递性网络的数量，物种数量"""
import pandas as pd
import numpy as np
from numpy import *

def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''筛选物种组合(矩阵，组合列表，Spearman列表)'''


def Select_Zuhe(CP_dic, Ass_lst, Sp_lst):
    Ass = []
    max_spear = 0
    max_index = 0
    for item in range(len(Sp_lst)):
        if Sp_lst[item][0] > max_spear:
            # print(max_spear,i)
            max_spear = Sp_lst[item][0]
            max_index = item
    if max_spear >= 0.6:
        C_mat = CP_dic[max_index][0]
        Ass = Ass_lst[max_index]
    else:
        C_mat = np.zeros((3, 3))
        max_spear = -0.15
    # print(C_mat,max_spear)
    return C_mat, Ass




if __name__=="__main__":
    path0 = "C:/Users/97899/Desktop/N/N_year/"
    path = "C:/Users/97899/Desktop/N/"
    str_df = pd.read_excel(path + "Network/Strong_index.xls")
    df_exp = pd.read_excel(path + "实验处理_ex.xls")
    str_df.set_index('Unnamed: 0')
    loop_spec=dict([k,[]] for k in range(1,39))
    chain_spec=dict([k,[]] for k in range(1,39))
    """获得物种系数"""
    for year in range(2008, 2009):
        print(year)
        # 获取物种组合
        path1 = path0 + "N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        # 获取Spearman值
        path3 = path0 + "N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        # 根据Spearman值筛选物种组合
        for ex in range(1, 39):
            ex = float(ex)
            # 环
            path2 = path0 + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
            CP_mat = LoadDict(path2)
            if str_df.loc[ex - 1, year] > 0:

                if year < 2016:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
                else:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
                loop_spec[ex].append(len(Ass))
            if str_df.loc[ex - 1, year] == 0:

                if year < 2016:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
                else:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
                chain_spec[ex].append(len(Ass))
    print(loop_spec,chain_spec)
    # print(pd.DataFrame(loop_spec))
    # print(pd.DataFrame(chain_spec))
    "网络中物种的平均数量、非传递性网络和传递性网络的数量"
    loop_num={}
    chain_num={}
    spec={"loop":{},"chain":{}}
    gb=df_exp.groupby("氮素")
    for g in gb:
        loop_num[g[0]]=0
        chain_num[g[0]]=0
        spec["loop"][g[0]]=0
        spec["chain"][g[0]]=0
        for key in g[1]["顺序"]:
            if key not in [1,20]:
                loop_num[g[0]]+=len(loop_spec[key])
                chain_num[g[0]]+=len(chain_spec[key])
                spec["loop"][g[0]]+=sum(loop_spec[key])
                spec["chain"][g[0]]+=sum(chain_spec[key])
        print(g[0],spec["loop"][g[0]]/loop_num[g[0]],spec["chain"][g[0]]/chain_num[g[0]])
    print(loop_num,chain_num)
    print(spec)