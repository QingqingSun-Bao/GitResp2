# -*- coding: utf-8 -*-
# @Time:2021/4/2420:40
# @File:Fig9_comp_bio_ram.py
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

# 载入字典
# def Load_dic(path):
#     f=open(path,encoding='utf-8')
#     dic=eval(f.read())
#     f.close()
#     return dic

def LoadDict(path):
    fr = open(path, encoding='gbk')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic

if __name__=="__main__":
    path="C:/Users/97899/Desktop/N/"
    spec_dic=LoadDict(path+"Attribute/bio_rem_comp.txt")
    # print(spec_dic)
    for key in spec_dic.keys():
        dic = dict((k, []) for k in ["bio", "ram_bio", "comp"])
        print(key)
        for item in spec_dic[key]:
            dic["bio"].append(item[2])
            dic["ram_bio"].append(item[3])
            dic["comp"].append(item[4])
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.scatter(dic["comp"],dic["bio"],label="Biomass")
        plt.scatter(dic["comp"],dic["ram_bio"],label="Ramets")
        plt.title(key)
        plt.legend()
        plt.xlabel("Competition coefficient")
        plt.show()
    # sorted([(x,y) for x,y in zip(dic["comp"],dic["bio"])])

