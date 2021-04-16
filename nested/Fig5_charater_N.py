# -*- coding: utf-8 -*-
# @Time:2021/4/1512:58
# @File:Fig5_charater_N.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import statsmodels.api as sm


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""标准化"""


def get_normalise(dic_chara):
    z = [dic_chara[key] for key in dic_chara.keys()]
    z_m=np.mean(z)
    z_sigma=np.std(z)
    for key in dic_chara.keys():
        dic_chara[key]=(dic_chara[key]-z_m)/z_sigma
    return dic_chara

"""获得欧氏距离"""
def get_Euclidean(dic_chara):
    dist = 0
    data = get_normalise(dic_chara)
    for key1 in data.keys():
        for key2 in data.keys():
            dist += (data[key1] - data[key2]) ** 2
    return dist/2


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_NODF = pd.read_excel(path + "Network/loop_NODF.xls")
    dic_attr=LoadDict(path+"Attribute/bio_rem.txt")
    """单独获取自然年份的数据"""
    df_nature=df_NODF[df_NODF["year"]==2008]
    print("nature",df_nature)
    bio_ = []
    rem_ = []
    for item in range(np.shape(df_nature)[0]):
        df_ex = df_nature.iloc[item, :]
        year = int(df_ex["year"])
        ex = df_ex["ex"]
        print(year, ex)
        biomass, remat = dic_attr[year][ex]
        # 计算物种的性状距离
        bio_.append(get_Euclidean(biomass))
        rem_.append(get_Euclidean(remat))


    """根据氮素分组并去除2008年"""
    gb = df_NODF.groupby("N")
    dic_N = {}
    for g in gb:
        bio_dist = []
        rem_dist = []
        for item in range(np.shape(g[1])[0]):
            df_ex = g[1].iloc[item, :]
            year = int(df_ex["year"])
            ex = df_ex["ex"]
            if float(year) > 2008.0:
                # 获得性状数据
                print("非自然年份",year, ex)
                biomass, remat = dic_attr[year][ex]
                # 计算物种的性状距离
                bio_dist.append(get_Euclidean(biomass))
                rem_dist.append(get_Euclidean(remat))
        dic_N[g[0]] = [bio_dist, rem_dist]

    """合并自然年份的数据"""
    dic_N[0][0].extend(bio_)
    dic_N[0][1].extend(rem_)
    print(dic_N)


    """画图"""
    labels=["0","1","2","3","5","10","15","20"]
    bio_mean=[]
    bio_std=[]
    rem_mean=[]
    rem_std=[]
    for key in dic_N.keys():
        # if float(key)<50.0:
            bio_mean.append(np.mean(dic_N[key][0]))
            bio_std.append(np.std(dic_N[key][0]))
            rem_mean.append(np.mean(dic_N[key][1]))
            rem_std.append(np.std(dic_N[key][1]))
    """构建回归"""
    y = bio_mean[:-1]
    print("y",y)
    m = 0.2
    x1 = [0, np.log10(1) + m, np.log10(2) + m, np.log10(3) + m, np.log10(5) + m, np.log10(10) + m,
          np.log10(15) + m, np.log10(20) + m]
    x2 = [[i, i ** 2] for i in x1]
    x = sm.add_constant(x2)
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    y_fitted = results.fittedvalues
    plt.plot(x1, y_fitted, "r")
    labels = x1
    err=bio_std[:-1]
    plt.bar(labels, y, width=0.1, yerr=err)
    plt.ylabel("MFD in biomass", fontdict={"size": 15})
    plt.xlabel("N addition rate", fontdict={"size": 15})
    plt.xticks(x1, ["0", "1", "2", "3", "5", "10", "15", "20"])
    plt.text(np.log10(10) + m, 50, r"$r^2=0.857**$")
    plt.show()

    plt.show()