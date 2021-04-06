# -*- coding: utf-8 -*-
# @Time:2021/4/110:31
# @File:Fig3_Spatial variability.py
"""物种分布的空间变异性08年的10块区域地"""
from sqlalchemy import create_engine
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt



if __name__=="__main__":
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    I=[]
    for year in range(2008,2009):
        data = pd.read_sql(str(year), con=engine)
        gb=data.groupby(["区组"])
        rich=[]
        for g in gb:
            df_rich=g[1]
            set_rich=set(df_rich["物种"])
            rich.append(len(set_rich))
        miu= mean(rich)
        sigma= std(rich)
        spatial_variability=(sigma*sigma)/(miu*miu)-1/miu+1
        I.append(spatial_variability)
    plt.scatter(linspace(2008, 2020, 13), I)
    plt.axhline(y=1, color='r', linestyle='-')
    plt.plot(linspace(2008,2020,13),I)
    plt.ylabel("Clumping")
    plt.xlabel("Year")
    plt.title("Nature")
    plt.show()

