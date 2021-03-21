import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sb
import math

df_Int = pd.read_excel("C:/Users/97899/Desktop/N/Intransitivity/Tao_C20.xls", sheet_name="Int")
df_wth = pd.read_excel("C:/Users/97899/Desktop/N/Enveriment/weather_rain_20.xls")
ind = np.linspace(2008, 2020, 13).tolist()
df_wth.set_index("Unnamed: 0", inplace=True)
Int = []
wth_rain = []
print(df_Int)
print(df_Int.columns)
# for jtem in ind:
#     A = [i for i in g[1].loc[:, int(year)] if 1 >= i > -0.15]  # 取非传递性水平
#     Int.extend(A)
#     # B = np.repeat(math.log(df_wth.iloc[i, 1],10), len(A))
#     B = np.repeat(df_wth.loc[jtem, 1], len(A))
#     print(jtem,B)
#     wth_rain.extend(B)
#     #     print(df_wth.iloc[i,1])
# print(np.max(wth_rain), np.min(wth_rain))
# print(len(wth_rain), len(Int))
# x = sm.add_constant(wth_rain)
# model = sm.OLS(Int, x)
# results = model.fit()
# print(results.summary())
# # y_fitted = results.fittedvalues
# # plt.scatter(wth_rain, Int)
# sb.regplot(wth_rain, Int, fit_reg=True, x_jitter=5, y_jitter=0.05, scatter_kws={"alpha": 1 / 3})
# # if dtem == "temp":
# #     plt.text(16, 0.8, r"$r^2=0.061***$", fontsize=13)
# #     plt.xlabel("Temperature Seasonality", fontsize=15)
# if dtem == "rain":
#     plt.text(300, 0.8, r"$r^2=0.036**$", fontsize=13)
#     plt.xlabel("Precipitation of 9-12 months", fontsize=15)
# plt.ylabel('Intransitivity level', fontsize=15)

# plt.figure(figsize=(10,4))
# x=np.linspace(2008, 2020, 13).tolist()
# y1=df_wth["season"].values
# plt.plot(x,y1,label="Precipitation Seasonality")
# y2=df_wth["all"].values-y1
# plt.plot(x,y2,label="Precipitation nonSeasonality")
# plt.xticks((2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020))
# # plt.xlim([2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])
# plt.legend()



plt.show()
