"""李雅普诺夫指数"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


path = 'C:/Users/97899/Desktop/N/'
df_bio = pd.read_excel(path + 'Biomass/biomass.xls')
df_site = pd.read_excel(path + '实验处理_site.xls')
columns = ['顺序', '氮素', '频率', '刈割']
df_bio = pd.concat([df_bio, pd.DataFrame(columns=columns)])

'''biomass and experiment deal'''

for item in range(df_site.shape[0]):
    for jtem in range(df_site.shape[0]):
        if int(df_bio.iloc[item, 0]) == int(df_site.iloc[jtem, 1]):
            df_bio.loc[item, '顺序'] = df_site.iloc[jtem, 2]
            df_bio.loc[item, '氮素'] = df_site.iloc[jtem, 3]
            df_bio.loc[item, '频率'] = df_site.iloc[jtem, 4]
            df_bio.loc[item, '刈割'] = df_site.iloc[jtem, 5]

'''Lyaponuv指数'''
mean_ls = []

g_nat = pd.DataFrame()
gb_N = df_bio.groupby('氮素')
# N1=[1]
N1=[0,1,2,3,5,10,15,20,50]
plt.figure()
l=1
for N in N1:
    lap_dic = {}
    for g in gb_N:
        if g[0] == N:
            g_nat = g[1].drop(['样地号', '顺序', '氮素', '频率', '刈割'], 1).mean()
    for g in gb_N:
        if g[0] != N:
            print()
            g_mean = g[1].drop(['样地号', '顺序', '氮素', '频率', '刈割'], 1).mean()
            A = pd.DataFrame(g_mean.values-g_nat.values)
            fx_dx = np.log(abs(A / int(g[0]-N)))
            lap_dic[g[0]] = fx_dx
    plt.subplot(3,3,l)
    x = np.linspace(2008,2020,13)
    x_new = np.linspace(x.min(), x.max(), 300)
    for key in lap_dic.keys():
        y_smooth = make_interp_spline(x, lap_dic[key])(x_new)
        plt.plot(x_new,y_smooth,label=("N=%d" % int(key)))
    plt.title("Lyaponuv Index N=%d"%N)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.4), ncol=2,fontsize=7)
    # plt.step(fontsize='xx-small')
    l=l+1
plt.show()

