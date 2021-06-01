# -*- coding: utf-8 -*-
# @Time:2021/4/110:31
# @File:Fig3_Spatial variability.py
"""物种分布的空间变异性08年的10块区域地"""
from sqlalchemy import create_engine
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    # 建立字典
    dic_var = dict([(k, []) for k in ["顺序", "氮素", "刈割", "频率"]])
    for key in dic_var.keys():
        I = []
        for year in range(2008, 2021):
            print(key,year)
            data = pd.read_sql(str(year), con=engine)
            gb = data.groupby([key])
            rich = []
            for g in gb:
                df_rich = g[1]
                set_rich = set(df_rich["物种"])
                rich.append(len(set_rich))
            miu = mean(rich)
            sigma = std(rich)
            spatial_variability = (sigma * sigma) / (miu * miu) - 1 / miu + 1
            I.append(spatial_variability)
        dic_var[key] = I
    print(dic_var)
    label = ["38 Experiments", "Nitrogen", "Mowing", "Frequency"]
    i = 0
    for key in dic_var.keys():
        plt.plot(linspace(2008, 2020, 13), dic_var[key],"o-",  label=label[i])
        i += 1
    plt.axhline(y=1, color='black', linestyle='-')
    # "38 Experimental treatments","Nitrogen","Mowing","Frequency"
    plt.ylabel("Clumping")
    plt.xlabel("Year")
    plt.legend()
    # plt.show()
    print(dic_var)
    pd.DataFrame(dic_var).to_excel("C:/Users/97899/Desktop/ans.xls")
