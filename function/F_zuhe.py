# -*- coding: utf-8 -*-
# @Time:2021/3/2415:12
# @File:F_zuhe.py
"""网络中的功能群"""
# 指定搜索路径
from Load_Save import LoadDict
from itertools import islice
from EX_Deal import ex_deal
import pandas as pd


def get_file(path_):
    with open(path_) as f:
        line_s = f.readlines()
    line_s = [[line.strip().split("\t")] for line in line_s[1:]]
    line = []
    for l in line_s:
        ls = []
        for l_tem in l[0]:
            if l_tem != "":
                ls.append(l_tem)
        line.append(ls)
    return line


def get_fun(plot_lst, df_f):
    comm_set = set()
    for index, item in enumerate(plot_lst):
        f = df_f[df_f["样地号"].values == str(int(eval(item)))]
        if index == 0:
            comm_set = set(f.loc[f.index[0],"功能群"].split("'"))
        else:
            comm_set = comm_set & set(f.loc[f.index[0],"功能群"].split("'"))

    single_lst = []
    for index, item in enumerate(plot_lst):
        f = df_f[df_f["样地号"].values == str(int(eval(item)))]
        single_lst.append(set(f.loc[f.index[0],"功能群"].split("'"))-comm_set-{"[", "]", ", "})
    single_set = single_lst
    comm_set = comm_set - {"[", "]", ", "}
    return comm_set,single_set


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/"
    zuhe_plot_dic = LoadDict(path + "N/Zuhe/Zuhe_plot20.txt")
    for year in range(2010, 2011):
        columns = ["样地号", "顺序", "功能群"]
        df_fun = pd.DataFrame(get_file(path + "Function/F_plot/" + str(year) + ".txt"), columns=columns)
        for ex in range(1, 39):
            comm,single=get_fun(zuhe_plot_dic[year][ex], df_fun)
            print("===============")
            print("%s共有的功能群"%str(ex),comm)
            print("单独存在的功能群",single)
