import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import stats

path = "C:/Users/97899/Desktop/N/Single/cao/ca_bio20_avg.xls"
df_huang_nested = pd.read_excel(path, sheet_name="ca_nested")
df_huang_nonested = pd.read_excel(path, sheet_name="ca_no_nested")
df_huang_nested.drop([0, 19], inplace=True)
df_huang_nonested.drop([0, 19], inplace=True)
Ind = list(np.linspace(2008, 2020, 13))

bio_hu = {}
avg_bio_nested = []
avg_bio_nonested = []
for year in Ind:
    bio_mean = np.mean([i for i in df_huang_nested.loc[:, int(year)] if i > 0])
    avg_bio_nested.append(bio_mean)
    bio_mean_nonested = np.mean([i for i in df_huang_nonested.loc[:, int(year)] if i > 0])
    avg_bio_nonested.append(bio_mean_nonested)
bio_hu["Loop"] = avg_bio_nested
bio_hu["Chained"] = avg_bio_nonested
# 趋势性检验
tau, P_value = stats.kendalltau(avg_bio_nested, avg_bio_nonested)
tau2, P_value2 = stats.kendalltau(avg_bio_nested[8:], avg_bio_nonested[8:])
tau1, P_value1 = stats.kendalltau(avg_bio_nested[:8], avg_bio_nonested[:8])
print(tau, P_value,tau2, P_value2,tau1, P_value1)
# 方差齐性检测
lev, P_val = stats.levene(avg_bio_nested, avg_bio_nonested)
T_statistic,T_pvalue=stats.ttest_ind(avg_bio_nested, avg_bio_nonested,equal_var=False)
print(T_statistic,T_pvalue)
plt.figure()
x = np.linspace(2008, 2020, 13)
x_new = np.linspace(x.min(), x.max(), 300)
# labels=["ring","Chained"]
for key in bio_hu.keys():
    print(bio_hu[key])
    y_smooth = make_interp_spline(x, bio_hu[key])(x_new)
    plt.scatter(x, bio_hu[key])
    plt.plot(x, bio_hu[key], label=("%s" % key))
    print(key)
plt.title("Cleistogenes squarrosa",fontdict={"size":15})
# 黄囊苔草Carex korshinskyi，糙隐子草Cleistogenes squarrosa，羊草Leymus chinensis,大针茅Stipa grandis P.A. Smirn.
# 羽茅 Achnatherum sibiricum
# plt.legend(loc='upper right', bbox_to_anchor=(0.3, 1), ncol=1, fontsize=7)
# plt.axvline(x=2015, ls="-", c="green")
# plt.text(2010, 20, "tau=%.3f**" % tau1, fontdict={"size": 13})
# plt.text(2016, 20, "tau=%.3f" % tau2, fontdict={"size": 13})
plt.text(2016, 12, "tau=%.3f*** "% tau, fontdict={"size": 13})
plt.xlabel("Year",fontdict={"size":15})
plt.ylabel("Average plot biomass(g)",fontdict={"size":15})
# Biomass of a plot (g)
# Weight per plant(g)
# Number of Ramets/Individuals
plt.show()
