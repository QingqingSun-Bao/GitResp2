# -*- coding: utf-8 -*-
# @Time:2021/5/1718:47
# @File:data_all_biomass.py
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

"""各物种组配所在样地的平均生物量"""
df_bio = pd.read_excel('C:/Users/97899/Desktop/N/Biomass/biomass.xls', sheet_name="bio_site")
df_ex = pd.read_excel('C:/Users/97899/Desktop/N/实验处理_site.xls')
df_bio.set_index(["site"], inplace=True)
ex_bio = {}
ind = np.linspace(2008, 2020, 13).tolist()
gb = df_ex.groupby(["顺序"])
for year in ind:
    ex_bio[year] = []
    for g in gb:
        site=set(g[1]["样地号"])
        sum_ = 0
        n=0
        for i in site:
            if df_bio.loc[float(i), year]!=0:
               sum_ = sum_ + df_bio.loc[float(i), year]
            else:
                n+=1
        ex_bio[year].append(sum_ / (len(site)-n))
print(ex_bio)

pd.DataFrame(ex_bio).to_excel('C:/Users/97899/Desktop/N/Biomass/bio_all.xls')
