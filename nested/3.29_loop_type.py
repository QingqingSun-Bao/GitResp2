# -*- coding: utf-8 -*-
# @Time:2021/3/2912:44
# @File:3.29_loop_type.py
"""分氮素讨论网络中loop的多样性"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy

path = "C:/Users/97899/Desktop/N"
df_loop_type = pd.read_excel(path + "/Network/loop_type.xls")
df_ex_loop = pd.read_excel(path + "/Network/all_loop_ex.xls")
df_loop_type["sum"] = df_loop_type.apply(lambda x: x["Short"] + x["Long"] +
                                                   x["Independent"] + x["Nested"]
                                                   + x["Cross"], axis=1)
"根据含有loop类型的多少进行分类"
gb = df_loop_type.groupby(["sum"])
for g in gb:
    print(g[0], g[1].shape[0])


# 合并两个数据框concat,axis=1,列数增加
df = pd.concat([df_ex_loop, df_loop_type["sum"]], axis=1)
# print(df.columns)
df_=df[~df["year"].isin([2008])]

"类型多少与复杂度的关系"
gb = df_.groupby(["N"])
ymean = []
ysd = []
for g in gb:
    g_1 = g[1].groupby(["sum"])
    y_mean = []
    y_sd = []
    for g1 in g_1:
        y_mean.append(np.mean(g1[1]["complexity"]))
        y_sd.append(np.std(g1[1]["complexity"]))
    ymean.append(y_mean)
    ysd.append(y_sd)
N = [0, 1, 2, 3, 5, 10, 15, 20, 50]
# for i in range(len(ymean)):
#     x=np.linspace(1,len(ymean[i]),len(ymean[i]))
#     plt.errorbar(x,ymean[i],ysd[i],fmt="o-",label=N[i],uplims=True, lolims=True)
#     # plt.plot(x_new,y_smooth)
# plt.legend()
# plt.xlabel("Number of loop type")
# plt.ylabel("Complexity")
# plt.show()

"类型与年份的关系"
gb = df.groupby(["year"])
x=np.linspace(2008,2021,13)
type_year=[]
com_year=[]
for g in gb:
    type_year.append(np.mean(g[1]["sum"]))
    com_year.append(np.mean(g[1]["complexity"]))
print(type_year)
x_new = np.linspace(x.min(), x.max(), 300)
plt.scatter(x,type_year)
plt.scatter(x,com_year)
y_smooth= make_interp_spline(x, type_year)(x_new)
y_smooth_comp= make_interp_spline(x, com_year)(x_new)

plt.plot(x_new,y_smooth,label="Average Type")
plt.plot(x_new,y_smooth_comp,label="Average complexity")
plt.ylabel("Number of loop type/Complexity")
plt.xlabel("Year")
plt.legend(bbox_to_anchor=(0.9, 1))
plt.show()


"类型与复杂度的关系随时间的变化"
for i in range(len(ymean)):
    x=np.linspace(1,len(ymean[i]),len(ymean[i]))

    # plt.plot(x_new,y_smooth)


"各年份loop类型的分布"