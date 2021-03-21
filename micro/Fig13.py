# -*- coding: utf-8 -*-
# @Time:2021/2/2511:46
# @File:Fig13.py
# @Software:PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path="C:/Users/97899/Desktop/N/"
df_huang=pd.read_excel(path+"Single/huang/hu_bio20_avg.xls")
df_complex=pd.read_excel(path+"Network/flow_hierarchy.xls")

# print("黄",df_huang)
# print("复杂",df_complex)

"""黄囊苔草与生物量的对应信息"""
ind=np.linspace(2008,2020,13)
hu_plex=[]
hu_chain=[]
for year in ind:
    for ex in range(0,38):
        if df_huang.loc[ex,year]!=0:
            if df_huang.loc[ex,year]==38.54333333333333:
                continue
            if df_complex.loc[ex,year]==0.6528571428571428:
                continue
            hu_plex.append([df_huang.loc[ex,year],df_complex.loc[ex,year]])
        if df_complex.loc[ex,year]==0:
            hu_chain.append([df_huang.loc[ex,year],df_complex.loc[ex,year]])

df_hu_plex=pd.DataFrame(hu_plex, columns=["huang","complex"])

df_hu_plex.sort_values("huang", inplace=True)


"""取均值"""
_mean=[]
ind_bio=np.linspace(5,40,8)
for j in ind_bio:
    X1=[]
    Y1=[]
    for i in range(np.shape(df_hu_plex)[0]):
        if j-5<=df_hu_plex.iloc[i,0]<j:
            X1.append(df_hu_plex.iloc[i,0])
            Y1.append(df_hu_plex.iloc[i,1])
    _mean.append([np.mean(X1),np.mean(Y1)])

df_mean=pd.DataFrame(_mean,columns=["X_mean","Y_mean"])
"""找最大点的合集"""

# df_hu_plex.sort_values("complex")
# max_bioss=[]
# for i in range(df_hu_plex.shape[0]):
#         for ii in range(df_hu_plex.shape[0]):
#             if df_hu_plex.loc[ii,jj]>df_hu_plex.loc[i,jj]:




"""每个复杂度的最大值"""
gb=df_hu_plex.groupby(["complex"])
X_,Y_=[],[]
for g in gb:
    X_.append(float(g[0]))
    Y_.append(np.max(g[1].loc[:,"huang"]))
    print(g[0],g[1])
print(X_,Y_)



"""画图"""
X_all=df_hu_plex.loc[:,"complex"].values
# print("黄囊苔草最大值",np.max(X))
Y_all=df_hu_plex.loc[:,"huang"].values
# plt.scatter(X_all,Y_all)
# plt.scatter(X_,Y_)
X_mean=df_mean.loc[:,"X_mean"].values
Y_mean=df_mean.loc[:,"Y_mean"].values
# plt.scatter(X_mean,Y_mean)
# plt.plot(X_mean,Y_mean,"r")

# plt.show()


from scipy.optimize import leastsq


# 待拟合的数据
X = np.array(X_)
Y=np.array(np.log(Y_))


# 二次函数的标准形式
# def func(params, x):
#     a, b, c = params
#     return a * x * x + b * x + c
def func(params, x):
    b, c = params
    return b * x + c

# 误差函数，即拟合曲线所求的值与实际值的差
def error(params, x, y):
    return func(params, x) - y


# 对参数求解
def slovePara():
    p0 = [10, 10]
    # p0=[10,10,10]
    Para = leastsq(error, p0, args=(X, Y))
    return Para


# 输出最后的结果
def solution():
    Para = slovePara()
    b, c = Para[0]
    print("b=",b," c=",c)

    print("cost:" + str(Para[1]))
    print("求解的曲线是:")
    # print("y="+str(round(a,2))+"x*x+"+str(round(b,2))+"x+"+str(c))
    print("y=exp("+ str(round(b, 2)) + "x)*" + str(np.exp(c)))
    # 准确度

    plt.figure(figsize=(8,6))
    plt.scatter(X_all, Y_all)#所有数据

    #   画拟合直线
    x=np.linspace(0,1,100) ##在0-15直接画100个连续点
    # y=a*x*x+b*x+c #函数式
    y=np.exp(b*x+c)
    plt.plot(x,y,color="black",linewidth=2)
    plt.xlabel("complex")
    plt.ylabel("Biomass of Carex xanthum")
    plt.title("Relationship between biomass and complexity of Carex xanthum ")
    # plt.legend() #绘制图例

    # r-squared
    print(len(y),len(Y_))
    yhat=[np.exp(b*x+c) for x in X_] #拟合值
    ybar=sum(Y_)/len(Y_)
    ssreg=0
    for i in range(len(yhat)):
        ssreg+=(yhat[i]-Y_[i])**2
    sstot=sum([(yi - ybar)**2 for yi in Y_])#真实值
    r2=1-ssreg/sstot
    print("r2",r2)
    print(ssreg)
    plt.show()

solution()



