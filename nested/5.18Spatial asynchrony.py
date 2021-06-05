# -*- coding: utf-8 -*-
# @Time:2021/5/1814:20
# @File:5.18Spatial asynchrony.py

"""计算物种的异步性"""
import pandas as pd
import numpy as np


if __name__ == "__main__":
    df_bio=pd.read_excel("C:/Users/97899/Desktop/N/Biomass/biomass.xls")
    df_ex=pd.read_excel("C:/Users/97899/Desktop/N/实验处理_site.xls")
    df_bio=df_bio.set_index(["site"])
    dic_asy = {}
    dic_tempdata={}
    gb=df_ex.groupby(["顺序"])
    for g in gb:
        dic_tempdata[g[0]]={}
        local_site=g[1]["样地号"]
        for s in local_site.values:
            dic_tempdata[g[0]][s]=df_bio.loc[s,:]
    # 计算方差协方差阵
    l_var = []
    l_cov = []
    for k in dic_tempdata.keys():
        l=[]
        for s in dic_tempdata[k].keys():
            l.append(dic_tempdata[k][s])
        mat_s=np.mat(l)
        cov=np.cov(mat_s)
        # 取local的方差
        w_ii=[np.sqrt(cov[i,i]) for i in range(len(l))]
        l_var.append(sum(w_ii)*sum(w_ii))
        # 两个path之间的协方差
        cov_sum=0
        # print(cov)
        for i in range(1,len(l)):
            for j in range(i+1,len(l)):
                cov_sum=cov_sum+cov[i,j]
        l_cov.append(cov_sum)
    lst=[]
    for ex in range(len(l_cov)):
        # print(l_cov[ex],l_var[ex])
        asy=1-l_cov[ex]/l_var[ex]
        dic_asy[ex]=asy
        lst.append(asy)
        print(asy)
    # pd.DataFrame(lst).to_excel("C:/Users/97899/Desktop/N/Network/spa_asy.xls")