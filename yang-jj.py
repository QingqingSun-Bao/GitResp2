# -*- coding: utf-8 -*-
# @Time:2021/3/99:15
# @File:yang-jj.py
# @Software:PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
df=pd.read_excel("C://Users/97899/Desktop/N/YJJ/生物量和多样性的抵抗力.xls")
print(df)
gb=df.groupby("年份")
Y=[]
for g in gb:
    # 生物量抵抗力（%）
    # 物种数抵抗力（%）
    Y.append(np.mean(g[1].loc[:,"物种数抵抗力（%）"]))
X=np.linspace(2009,2020,12)
# plt.subplots(4,3,1)
plt.scatter(X,Y)
x_new = np.linspace(np.min(X), np.max(X), 300)
y_smooth = make_interp_spline(X, Y)(x_new)
plt.plot(x_new,y_smooth)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlim(12,[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
plt.xlabel("年份")
plt.ylabel("物种数抵抗力（%）")

plt.show()