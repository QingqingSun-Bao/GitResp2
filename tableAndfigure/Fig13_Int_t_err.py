"""非传递性在不同氮素下的时间变化规律——时序图"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from matplotlib.pyplot import MultipleLocator

'''各种处理下非传递性的时间变化规律图'''

path = 'C:/Users/97899/Desktop/N/Intransitivity/Tao_C20.xls'
path1 = 'C:/Users/97899/Desktop/N/实验处理_ex.xls'
df_Int = pd.read_excel(path, sheet_name='Int')
df_ex = pd.read_excel(path1)
columns = ['顺序', '氮素', '频率', '刈割']
df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])

'''Intransitivity and experiment deal'''

for item in range(df_Int.shape[0]):
    for jtem in range(df_ex.shape[0]):
        if int(df_Int.iloc[item, 0]) + 1 == int(df_ex.iloc[jtem, 1]):
            df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
            df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
            df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
            df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
N_0=df_Int.loc[[0,19],:]
print("N_0",N_0)
df_Int.drop([0, 19], inplace=True)

'''去掉纯自然状态'''
l = 1
ind = np.linspace(2009, 2020, 12).tolist()
gb = df_Int.groupby("频率")
# 按氮素将传递性进行分组
plt.figure()
# R2=['0.215***','0.340***','0.165***','0.112*','0.079','0.024','0.002','001','0.045']
#R2 = ['0.081*', '0.227***', '0.118*', '0.128*', '0.098*', '0.000', '0.000', '0.000', '0.000']
# R2=[0.263,0.219]
R2=[0.253,0.154]
for g in gb:
    te = []
    x1 = []
    err=[]
    for item in ind:
        A = [i for i in g[1][item].values if -0.15 < i < 1]
        te.append(np.mean(A))
        err.append(np.std(A,ddof=1))
        x1.append((item - 2008))
        if g[0] == 50:
            print(item, A)
    X = sm.add_constant(x1)
    te = np.array(te)
    model = sm.OLS(te, X).fit()
    print("第%d个" % l)
    print(model.summary())
    y_fitted = model.fittedvalues
    plt.subplot(3, 3, l)
    # plt.title("N=%s"r"$(g N m^{-2} year^{-2})$" % str(int(g[0])), fontsize=11)
    plt.title("F=%s" % str(int(g[0])), fontsize=11)
    # plt.scatter(x1, te, c='b', s=50, marker='o')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xticklabels(['2010', '2012', '2014', '2016', '2018', '2020'])
    ax.errorbar(x1,te,err,fmt="s",capsize=3)
    plt.ylim(0, 1.0)
    if l<=5:
       plt.plot(x1, y_fitted, 'r')
       plt.text(2, 0.8, r"$r^2=%s$" % R2[l - 1], fontsize=13)
    if l == 4:
        plt.ylabel("Intransitivity level", fontsize=20)
    if l == 8:
        plt.xlabel("Time", fontsize=20)
    l = l + 1
plt.show()

'''纯自然状态下非传递性的变化'''
plt.figure()
te=[]
err=[]
x1=[]
for item in ind:
    A = [i for i in N_0[item].values if -0.15 < i < 1]
    te.append(np.mean(A))
    err.append(np.std(A, ddof=1))
    x1.append((item - 2008))
X = sm.add_constant(x1)
te = np.array(te)
model = sm.OLS(te, X).fit()
print(model.summary())
y_fitted = model.fittedvalues
plt.title("Nature", fontsize=11)
# plt.scatter(x1,te)
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_xticklabels(['2010', '2012', '2014', '2016', '2018', '2020'])
ax.errorbar(x1,te,err,fmt="s",capsize=3)
plt.ylim(0, 1.0)
plt.ylabel("Intransitivity level", fontsize=15)
plt.xlabel("Time", fontsize=15)
plt.text(2, 0.8, r"$r^2=%.3f$,P=0.511" %(0.026), fontsize=13)
plt.plot(x1, y_fitted, 'r')
plt.show()