import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sb
import math

df_Int = pd.read_excel("C:/Users/97899/Desktop/N/Intransitivity/Tao_C20.xls", sheet_name="Int")
path1 = 'C:/Users/97899/Desktop/N/实验处理_ex.xls'
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
N_0 = df_Int.loc[[1,2,21,20], :]
N_1 = df_Int.loc[[0,19], :]
#1,2,21,20
# df_Int.drop([0, 19], inplace=True)

gb = df_Int.groupby(["氮素"])
ind = np.linspace(2009, 2020, 12).tolist()
# weather = ["rain", "temp"]
dtem = "rain_"
df_wth = pd.read_excel("C:/Users/97899/Desktop/N/Enveriment/weather_rain_20.xls")
df_wth.set_index("Unnamed: 0", inplace=True)
# df_temp = pd.read_excel("C:/Users/97899/Desktop/N/Enveriment/weather_temp_.xls")
# df_temp.set_index("Unnamed: 0", inplace=True)
print(df_wth)
Int = []
wth_rain = []
category = []
for g in gb:
    for year in ind:
        A = [i for i in g[1].loc[:, int(year)] if 1 >=i > -0.15]  # 取非传递性水平
        Int.extend(A)
        B = np.repeat(math.log(df_wth.loc[year-1,"9"]+df_wth.loc[year,"7"],10), len(A)) # 取年份
        # season  avg_sea    all  avg_all      7  avg_7
        # /(df_temp.loc[int(year), 1]+10)
        wth_rain.extend(B)
        category.extend(np.repeat(g[0], len(A)))  # 各个氮素水水平
df_cate = pd.DataFrame([Int, wth_rain, category]).T
df_cate.columns = ["Int", "rain", "cate"]
x = sm.add_constant(wth_rain)
model = sm.OLS(Int, x)
results = model.fit()
print(results.summary())
print(results.rsquared)
# y_fitted = results.fittedvalues
N = [0, 1, 2, 3, 5, 10, 15, 20, 50]
color=["red","black","sienna",
       "gold","lawngreen","green",
       "teal","blue","fuchsia"]
gbc = df_cate.groupby("cate")
# for i,color in enumerate(zip(N,colors)):

for index,g in enumerate(gbc):
    plt.scatter(g[1].loc[:, "rain"], g[1].loc[:, "Int"],marker="o",color="white",facecolor="none"
                ,edgecolors=color[index],s=90)
plt.legend(ncol=1, bbox_to_anchor=(1.1,0.5), fontsize=8,labels=N)
sb.regplot(wth_rain, Int, fit_reg=True, scatter_kws={"alpha": 0})

plt.xlabel("Precipitation of Sep-Dec", fontsize=15)
# Growing Season Precipitation,1-7 months Precipitation
plt.text(1.45, 0.9, r"$r^2=%.3f$***"%(results.rsquared), fontsize=13)
plt.ylabel('Intransitivity level', fontsize=15)

plt.show()

'''纯自然状态下'''
# Int_0=[]
# wth_rain_0=[]
# for year in ind:
#     A = [i for i in N_1.loc[:, int(year)] if 1 >= i > -0.15]  # 取非传递性水平
#     Int_0.extend(A)
#     B = np.repeat(math.log(df_wth.loc[year, "7"],10), len(A))  # 取年份
#     # season  avg_sea    all  avg_all      7  avg_7
#     wth_rain_0.extend(B)
# x = sm.add_constant(wth_rain_0)
# model = sm.OLS(Int_0, x)
# results = model.fit()
# print(results.summary())
# plt.scatter(wth_rain_0,Int_0,marker="o",color="b",s=90)
# sb.regplot(wth_rain_0,Int_0, fit_reg=True, scatter_kws={"alpha": 0})
# plt.xlabel("Precipitation of Jan-Jul ", fontsize=15)
# # Growing Season/Annual/
# plt.ylabel('Intransitivity level', fontsize=15)
# plt.ylim((0,1))
# plt.text(2.45,0.8,r"$r^2=0.001$")
# #
# plt.show()