# -*- coding: utf-8 -*-
# @Time:2021/5/1715:54
# @File:5.17 species asynchrony.py
"""计算物种的异步性"""
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

"""计算具体物种的方差"""


# def SelectSpec(engine, year, ex):
#     data = pd.read_sql(str(year), con=engine)
#     data_ex = data[data['顺序'] == ex]
#     spec = set(data_ex["物种"])
#     # print(spec)
#     sp_sigma = []
#     dic_sp = dict([k, []] for k in spec)
#     gb = data_ex.groupby(["样地号"])
#     for g in gb:
#         for i in range(np.shape(g[1])[0]):
#             data_spec = g[1].iloc[i, :]  # 把物种提出
#             dic_sp[data_spec["物种"]].append(float(data_spec["干重g"]))
#     # 计算所有物种的sigma
#     for k in dic_sp.keys():
#         sp_sigma.append(np.std(dic_sp[k]))
#
#     return np.sum(sp_sigma)


def SelectSpec(engine, ex):
    dic_sp = {}
    sp_sigma = []
    for year in range(2008, 2021):
        data = pd.read_sql(str(year), con=engine)
        data_ex = data[data['顺序'] == ex]
        spec = set(data_ex["物种"])
        for sp in spec:
            data_sp = data_ex[data_ex["物种"] == sp]
            sp_i = [float(i) for i in data_sp["干重g"]]
            if sp in dic_sp.keys():
                dic_sp[sp].append(np.mean(sp_i))
            else:
                dic_sp[sp] = [np.mean(sp_i)]

    # 计算所有物种的年际sigma
    for k in dic_sp.keys():
        if len(dic_sp[k]) < 13:
            for i in range(13 - len(dic_sp[k])):
                dic_sp[k].extend([0])
            sp_sigma.append(np.std(dic_sp[k]))
        else:
            sp_sigma.append(np.std(dic_sp[k]))

    return np.sum(sp_sigma)


"""计算生物量在时间上的方差"""

if __name__ == "__main__":
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    bio_df = pd.read_excel("C:/Users/97899/Desktop/N/Biomass/bio_all.xls")
    dic_asy = {}
    for ex in range(1, 39):
        ex = str(float(ex))
        sigma_sum = SelectSpec(engine, ex)
        print(ex, sigma_sum*sigma_sum)
        dic_asy[ex]=sigma_sum * sigma_sum
    # 群落水平上的方差
    t_sigma = []
    for i in range(38):
        std = np.std(bio_df.iloc[i, :])
        t_sigma.append(std * std)
    index = 0
    # dic_asy={'1.0': 32065.381558726975, '2.0': 21556.486017256393, '3.0': 18676.0999129408, '4.0': 20144.669311660313}
    print(dic_asy)
    for k in dic_asy.keys():
        print(1 - (t_sigma[int(float(k))-1] / dic_asy[k]))
        index += 1
