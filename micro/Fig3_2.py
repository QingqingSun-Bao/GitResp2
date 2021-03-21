# -*- coding: utf-8 -*-
# @Time:2021/2/1710:31
# @File:Fig3_2.py
# @Software:PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import stats
import statsmodels.api as sm

path = "C:/Users/97899/Desktop/N/"
rain = pd.read_excel(path + "Enveriment/weather_rain_20.xls")
rain.set_index(['Unnamed: 0'], inplace=True)
temp = pd.read_excel(path + "Enveriment/weather_temp_20.xls")
temp.set_index(['Unnamed: 0'], inplace=True)
path1 = path + 'Single/da/da_bio_rem.xls'

df_huang_nested = pd.read_excel(path1, sheet_name="da_loop_biorem")
df_huang_nonested = pd.read_excel(path1, sheet_name="da_chain_biorem")

title="Stipa grandis"
# 黄囊苔草Carex korshinskyi，糙隐子草Cleistogenes squarrosa，羊草Leymus chinensis,大针茅Stipa grandis P.A. Smirn.
ytitle="Number of Ramets/individuals"
# Biomass per plot(g),Weight per plant(g),Number of Ramets/individuals
df_huang_nested.drop([0, 19], inplace=True)
df_huang_nonested.drop([0, 19], inplace=True)
Ind = list(np.linspace(2008, 2020, 13))

bio_hu = {}
avg_bio_nested = []
avg_bio_nonested = []
weather_rain = []
weather_temp = []
for year in Ind:
    bio_mean = np.mean([i for i in df_huang_nested.loc[:, int(year)] if i > 0])
    avg_bio_nested.append(bio_mean)
    bio_mean_nonested = np.mean([i for i in df_huang_nonested.loc[:, int(year)] if i > 0])
    avg_bio_nonested.append(bio_mean_nonested)
    weather_temp.append(temp.loc[year, 1])
    weather_rain.append(rain.loc[year, "avg_sea"])
bio_hu["Loop"] = avg_bio_nested
bio_hu["chain"] = avg_bio_nonested
bio_hu["rain"] = weather_rain
bio_hu["temp"] = weather_temp
print(bio_hu)

'''建立温度与降雨量的回归'''
te = np.array(bio_hu["Loop"])

X = sm.add_constant(bio_hu["rain"])
model = sm.OLS(te, X).fit()
y_fitted = model.fittedvalues

te = te - y_fitted
X = sm.add_constant(bio_hu["temp"])
model = sm.OLS(te, X).fit()
y_fitted = model.fittedvalues
avg_bio_nested = te - y_fitted
bio_hu["loop"]=avg_bio_nested


te = np.array(bio_hu["chain"])
X = sm.add_constant(bio_hu["rain"])
model = sm.OLS(te, X).fit()
y_fitted = model.fittedvalues

te = te - y_fitted
X = sm.add_constant(bio_hu["temp"])
model = sm.OLS(te, X).fit()
y_fitted = model.fittedvalues
avg_bio_nonested = te - y_fitted
bio_hu["chain"]=avg_bio_nonested


# 趋势性检验
tau, P_value = stats.kendalltau(avg_bio_nested, avg_bio_nonested)
tau2, P_value2 = stats.kendalltau(avg_bio_nested[8:], avg_bio_nonested[8:])
tau1, P_value1 = stats.kendalltau(avg_bio_nested[:8], avg_bio_nonested[:8])
print("趋势检测", tau, P_value)
print("趋势检测1", tau1, P_value1)
print("趋势检测2", tau2, P_value2)
# 方差齐性检测
lev, P_val = stats.levene(avg_bio_nested, avg_bio_nonested)
T_statistic, T_pvalue = stats.ttest_ind(avg_bio_nested, avg_bio_nonested, equal_var=False)
print("T检测", T_statistic, T_pvalue)
plt.figure()
x = np.linspace(2008, 2020, 13)
x_new = np.linspace(x.min(), x.max(), 300)
keys=["loop","chain"]
for key in keys:
    y_smooth = make_interp_spline(x, bio_hu[key])(x_new)
    plt.scatter(x, bio_hu[key])
    plt.plot(x_new, y_smooth, label=("%s" % key))
plt.title(title)
plt.legend(loc='upper right', bbox_to_anchor=(0.2, 1), ncol=1, fontsize=7)
# plt.axvline(x=2015, ls="-", c="green")
# plt.text(2010, 0.3, "tau=%.3f**" % tau1, fontdict={"size": 9})
# plt.text(2020, 0.3, "tau=%.3f" % tau2, fontdict={"size": 9})
plt.text(2016, 2, "tau=%.3f*" % tau, fontdict={"size": 13})
plt.xlabel("Time")
plt.ylabel(ytitle)
# 羊草,2016,500,"tau=%.3f***"%tau
# 黄囊苔草，2016,100,"tau=%.3f"%tau
# 糙隐子草，2016,50,"tau=%.3f**"%tau
# 大针茅,2008,600,"tau=%.3f**"%tau
plt.show()
