# -*- coding: utf-8 -*-
# @Time:2021/6/39:46
# @File:6.3 Fig_coff_distrubution.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    dic_matic = LoadDict(path + "Competition/dic_comp.txt")
    df_ex.drop([0, 19], inplace=True)
    gb = df_ex.groupby("氮素")
    N=[0,1,2,3,5,10,15,20,50]
    N2=[]
    for i in range(9):
        for j in range(i+1,9):
            plt.figure()
            for g in gb:
                if float(g[0])==N[i] or float(g[0])==N[j]:
                    comp = []
                    for ex in g[1]["顺序"]:
                        for item in dic_matic[ex]:
                            comp.extend(item)
                    # plt.hist(comp)
                    sns.kdeplot(comp,shade=False,label=g[0])
            plt.legend()
            plt.title("all")
            # plt.xlim([np.linspace(0,1,10)])
            plt.xlabel("Competition Coefficient")
            plt.xticks([i * 0.1 for i in range(0, 11)])
            # [i * 0.1 for i in range(0, 11)]
            # 保存到本地
            # plt.show()
            plt.savefig(path+'Figure3/distribution/loop/all/'+'{}-{}.png'.format(N[i],N[j]))
