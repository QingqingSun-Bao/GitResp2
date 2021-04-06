# -*- coding: utf-8 -*-
# @Time:2021/4/614:12
# @File:data_remats.py
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
# engine=create_engine("mysql+pymysql://root:root@localhost:3306/nchenjiang?charset=utf8")
engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')

'''每年各个样地中的生物量'''


def Loaddic(path):
    fr = open(path, encoding='UTF-8')
    # 'unicode_escape'
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""各物种组配下的生物量"""
Zuhe_remats = Loaddic("C:/Users/97899/Desktop/N/Zuhe/Zuhe_plot20.txt")
ex_remats = {}
ind = np.linspace(2008, 2020, 13).tolist()
for year in ind:
    ex_remats[year] = {}
    df_remats = pd.read_sql(str(int(year)), con=engine)
    for key in Zuhe_remats[year].keys():
        zhu=0
        data = df_remats[df_remats["顺序"] == str(float(key))]
        for zh in Zuhe_remats[year][key]:
            dt1 = data[data["样地号"] == zh]
            dt3 = [0 if i is None else float(i) for i in dt1.loc[:,"株丛数"].values]
            zhu = zhu + sum(dt3)
        ex_remats[year][key] = zhu / len(Zuhe_remats[year][key])

pd.DataFrame(ex_remats).to_excel('C:/Users/97899/Desktop/N/Biomass/ramets_ex21.xls')
