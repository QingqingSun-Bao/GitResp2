# -*- coding: utf-8 -*-
# @Time:2021/3/2615:47
# @File:3.26_All_loop_ex.py
"""
所有的loop-network写入文件，
实验条件，年份，complexity
"""
import pandas as pd
import numpy as np


columns=["year","ex","complexity","N","Fre","Mow","temper","rain"]
path="C:/Users/97899/Desktop/"

df_ex=pd.read_excel(path+"N/实验处理_ex.xls")
df_complex=pd.read_excel(path+"N/Network/Strong_index.xls")
df_temper=pd.read_excel(path+"N/Enveriment/weather_temp_20.xls")
df_rain=pd.read_excel(path+"N/Enveriment/weather_rain_20.xls")

lst=[]
for year in range(2008,2021):
    df_cir=pd.read_excel(path+"N/Network/circle21.xls",sheet_name=str(year))
    for i in range(df_cir.shape[0]):
        if df_cir.iloc[i,1]>0:
            lst.append([year,i+1,df_complex.loc[i,year],
                        df_ex.loc[i,"氮素"],df_ex.loc[i,"频率"],df_ex.loc[i,"刈割"],
                        df_temper.loc[year-2008,0],df_rain.loc[year-2008,"season"]])
print(pd.DataFrame(lst,columns=columns))
pd.DataFrame(lst,columns=columns).to_excel(path+"N/Network/all_loop_ex.xls")



