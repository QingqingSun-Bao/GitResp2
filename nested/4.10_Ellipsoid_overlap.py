# -*- coding: utf-8 -*-
# @Time:2021/4/915:27
# @File:4.10_Ellipsoid_overlap.py
import pandas as pd
from numpy import *
from sqlalchemy import create_engine


def V_EnvironmentSite(tab):
    Dic={}
    gb=tab.groupby(["区组"])
    for g in gb:
        if int(float(g[0]))==1:
            Dic["site"]=[]
            Dic["N"]=[]
            Dic["M"]=[]
            Dic["F"]=[]
            for item in set(g[1]["样地号"]):
                Dic["site"].append(item)
                Dic["N"].append(list(set(g[1][g[1]["样地号"]==item]["氮素"]))[0])
                Dic["M"].append(list(set(g[1][g[1]["样地号"] == item]["刈割"]))[0])
                Dic["F"].append(list(set(g[1][g[1]["样地号"] == item]["频率"]))[0])
    V=pd.DataFrame(Dic,columns=["site","N","M","F"])
    return V

def M_Specisesite(tab):
    Dic = {}
    gb = tab.groupby(["区组"])
    for g in gb:

        if int(float(g[0])) == 1:
            Dic["site"] = []
            Dic["N"] = []
            Dic["M"] = []
            Dic["F"] = []
            for item in set(g[1]["样地号"]):
                Dic["site"].append(item)
                Dic["N"].append(list(set(g[1][g[1]["样地号"] == item]["氮素"]))[0])
                Dic["M"].append(list(set(g[1][g[1]["样地号"] == item]["刈割"]))[0])
                Dic["F"].append(list(set(g[1][g[1]["样地号"] == item]["频率"]))[0])
    M = pd.DataFrame(Dic, columns=["site", "N", "M", "F"])
    return M









if __name__ == "__main__":
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    df_ex = pd.read_excel("C:/Users/97899/Desktop/N/实验处理_site.xls")
    df = pd.read_sql(str(2008), con=engine)
    """构建一个环境变量矩阵V，Environment和site，并标准化"""
    V_matic=V_EnvironmentSite(df)




    """构建一个物种有无矩阵C"""
