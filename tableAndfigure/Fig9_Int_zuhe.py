"""非传递性与物种组配后的物种数——散点图"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm

path = 'C:/Users/97899/Desktop/U/Intransitivity/Tao_C.xls'
df_Int = pd.read_excel(path, sheet_name='Int')
df_zuhe = pd.read_excel(path, sheet_name='zuhe')
ind = np.linspace(2009, 2019,11).tolist()
x = []
y = []
print(df_Int.columns)
for item in range(38):
    for jtem in ind:
        if -0.15<df_Int.loc[item, jtem]<=1:
            y.append(df_Int.loc[item, jtem])
            x.append(df_zuhe.loc[item, jtem])
plt.figure()
X=sm.add_constant(x)
model=sm.OLS(y,X).fit()
print(model.summary())
plt.scatter(x,y,alpha=0.2)
sb.regplot(x,y,fit_reg=True,x_jitter=0.4,y_jitter=0.05,scatter_kws={"alpha":1/3})
# plt.text(,,r"$0.018**$")
plt.xlabel('The number of species in a Assembly')
plt.ylabel('Intransityvity Level')
plt.show()
