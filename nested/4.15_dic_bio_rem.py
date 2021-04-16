# -*- coding: utf-8 -*-
# @Time:2021/4/1521:03
# @File:4.15_dic_bio_rem.py
# -*- coding: utf-8 -*-
# @Time:2021/4/1512:58
# @File:4.15_charater_N.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic

def Savedict(path,datadict):
    file = open(path, "w",encoding='utf8')
    file.write(str(datadict))
    file.close()

"""组合中物种的生物量"""
def get_biomass(y, e, zh, zh_plot):
    df = pd.read_sql(str(y), con=engine)
    df = df[df["顺序"] == str(e)]
    dic_spe_bio = dict.fromkeys(zh, 0)
    dic_spe_remat = dict.fromkeys(zh, 0)
    for i_plot in zh_plot:
        df_site=df
        df_site= df_site[df_site["样地号"] == i_plot]
        spec = set(df_site["物种"])
        for j_spe in spec:
            if j_spe in zh:
                df_spec = df_site[df_site["物种"] == j_spe]
                for j_bio in df_spec.loc[:, "干重g"]:
                    dic_spe_bio[j_spe] += float(j_bio)
                for j_rem in df_spec.loc[:, "株丛数"]:
                    dic_spe_remat[j_spe] += float(j_rem)
    return dic_spe_bio, dic_spe_remat


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

# def get_feature(g):
#     for item in range(np.shape(g)[0]):
#         df_ex = g.iloc[item, :]
#         year = int(df_ex["year"])
#         ex = df_ex["ex"]
#         print(year, ex)
#         if float(year)>2008.0:
#             break
#         # 找到网络中的物种
#         get_amb = zuhe[year][ex]
#         # 找到物种组合所在的样地
#         get_plot = zuhe_plot[year][ex]
#         # 计算物种的生物量指标
#         biomass, remat = get_biomass(year, ex, get_amb, get_plot)
#         # 计算物种的性状距离
#         bio_dist.append(get_Euclidean(biomass))
#         rem_dist.append(get_Euclidean(remat))
#     return [bio_dist, rem_dist]


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_NODF = pd.read_excel(path + "Network/loop_NODF.xls")
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    df_loop_type=pd.read_excel(path + "Network/loop_type.xls")
    df_NODF["short"]=df_loop_type["Short"]
    df_NODF["nest"] = df_loop_type["Nested"]
    gb = df_NODF.groupby("year")
    dic={}
    for g in gb:
        dic[g[0]]={}
        for item in range(np.shape(g[1])[0]):
            df_ex = g[1].iloc[item, :]
            year = int(df_ex["year"])
            ex = df_ex["ex"]
            print(year, ex)
            # if float(year)>2008.0:
            #     break
            # 找到网络中的物种
            get_amb = zuhe[year][ex]
            # 找到物种组合所在的样地
            get_plot = zuhe_plot[year][ex]
            # 计算物种的生物量指标
            biomass, remat = get_biomass(year, ex, get_amb, get_plot)
            # 计算物种的性状距离
            dic[g[0]][ex]=[biomass,remat]
    Savedict(path+"Attribute/bio_rem.txt",dic)






    # for g in gb:
    #     bio_dist = []
    #     rem_dist = []
    #     gb_looptype=g[1].groupby("short")
    #     for gb_ in gb_looptype:
    #         """判断一下是short还是nest"""
    #         if int(gb_[0])==0:
    #             dic_short[g[0]]=get_feature(gb_[1])
    #         else:
    #             dic_nest[g[0]] = get_feature(gb_[1])

    # print(dic_nest)
    # print(dic_short)
    #
    # """画图"""
    # fig, ax = plt.subplots()
    # labels=["0","1","2","3","5","10","15","20","50"]
    # bio_mean=[]
    # bio_std=[]
    # rem_mean=[]
    # rem_std=[]
    # for key in dic_short.keys():
    #     bio_mean.append(np.mean(dic_short[key][0]))
    #     bio_std.append(np.std(dic_short[key][0]))
    #     rem_mean.append(np.mean(dic_short[key][1]))
    #     rem_std.append(np.std(dic_short[key][1]))
    # width = 0.35
    # ax.bar(labels, bio_mean, width, yerr=bio_std, label='biomass')
    #
    # ax.set_ylabel('Scores')
    # ax.set_title('short')
    # ax.legend()
    #
    # plt.show()