# -*- coding: utf-8 -*-
# @Time:2021/6/312:58
# @File:6.3 get_df_net_coff.py
import pandas as pd


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    df_ex.drop([0, 19], inplace=True)
    dic_matic = LoadDict(path + "Competition/dic_fre.txt")
    dic_N = dict([(k, []) for k in range(1,39)])
    # dic_N = dict([(k, []) for k in [0, 1, 2, 3, 5, 10, 15, 20, 50]])
    # gb = df_ex.groupby("氮素")
    #     # for g in gb:
    #     #     ex = g[1]["顺序"]
    #     #     lst = [0 for i in range(12)]
    #     #     for key in ex:
    #     #         for item in dic_matic[key]:
    #     #             lst = [i + j for i, j in zip(lst, item)]
    #     #     dic_N[g[0]] = lst
    for key in dic_matic.keys():
        lst = [0 for i in range(12)]
        for item in dic_matic[key]:
            lst = [i + j for i, j in zip(lst, item)]
        dic_N[key]=lst
        # 写入对应实验
        #
    # print(dic_matic)

    pd.DataFrame(dic_N).T.to_excel(path+"Competition/comp_fre.xls")
    # print(pd.DataFrame(dic_N).T)
    # .to_excel(path+"Competition/comp_fre.xls")
