# -*- coding: utf-8 -*-
# @Time:2021/4/2319:49
# @File:4.23_bio_rem_comp.py
"""记录物种，竞争系数，生物量和株丛数的数据"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from numpy import *


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def Save_dic(path, dic):
    file = open(path,"w")
    file.write(str(dic))
    file.close()


"""获取物种的竞争系数"""


def get_compete(c_mat):
    row_mean = []
    n = np.shape(c_mat)[0]
    for i in range(n):
        row_mean.append(np.mean(c_mat[i, :]))
    # 返回整个矩阵所有物种的竞争系数
    return row_mean


"""获得物种的样地信息"""


# 参数：物种字典，表，所在样地，所要考察的物种
def get_biomass_ramets(dic_s, table, plot, spec_set, year, ex):
    dic = dict((k, []) for k in spec_set)
    n=len(plot)
    for p in plot:
        spec_p = table[table["样地号"] == p]
        for i_sp in spec_set:
            if not dic[i_sp]:
                sp_att = spec_p[spec_p["物种"] == i_sp]
                dic[i_sp].append([float(sp_att["干重g"]), float(sp_att["株丛数"])])
            elif dic[i_sp]:
                sp_att = spec_p[spec_p["物种"] == i_sp]
                dic[i_sp][0][0] += float(sp_att["干重g"])
                dic[i_sp][0][1] += float(sp_att["株丛数"])
    for sp in spec_set:
        if sp in dic_s.keys():
            dic_s[sp].append([year, ex, dic[sp][0][0]/n, dic[sp][0][1]/n])
    return dic_s


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    # 找到loop所在的实验以及年份
    df_loop = pd.read_excel(path + "Network/loop_NODF.xls")
    # 找到loop所在的试验样地
    plot_dic = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    zuhe_dic = LoadDict(path + "Zuhe/Zuhe_20.txt")
    spec = ["大针茅", "羊草", "羽茅", '冰草', '黄囊苔草', '糙隐子草', '灰绿藜', '洽草', '猪毛蒿', '猪毛菜', '细叶韭',
            '轴藜', '小花花旗竿', '刺藜', '硬质早熟禾', '细叶鸢尾', '砂韭', '星毛委陵菜']
    dic_spec = dict((k, []) for k in spec)
    sum_plot=0
    for i in range(np.shape(df_loop)[0]):
        df_ex = df_loop.iloc[i, :]
        year = df_ex["year"]
        ex = df_ex["ex"]
        print(year, ex)
        table_df = pd.read_sql(str(int(year)), con=engine)
        table_df = table_df[table_df["顺序"] == str(ex)]  # 取出相应的试验区组
        dic_spec = get_biomass_ramets(dic_spec, table_df, plot_dic[year][ex], zuhe_dic[year][ex], year, ex)
        sum_plot+=len(plot_dic[year][ex])
        # 获取物种的竞争系数
        dic_matic = LoadDict(path + "N_pure/N_" + str(int(year)) + "/Cmat.txt")
        comp = get_compete(dic_matic[ex])
        for sp, comp_ in zip(zuhe_dic[year][ex], comp):
            if sp in dic_spec.keys():
                dic_spec[sp][-1].append(comp_)
    Save_dic(path+"Attribute/bio_rem_comp.txt",dic_spec)
    print(sum_plot)

    # 分物种记录其所在实验处理、竞争系数、生物量、株丛数
