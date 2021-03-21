"""Spearman值与物种数选取之间的关系"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as sm

path = 'C:/Users/97899/Desktop/U/Intransitivity/Tao_C.xls'
df_Sp= pd.read_excel(path,sheet_name='Spearman')
df_specise=pd.read_excel(path,sheet_name='zuhe')
ind=np.linspace(2009,2019,11).tolist()
x_specise=[]
y_spearman=[]
for j in ind:
    for i in df_Sp.index:
        if -0.15<df_Sp.loc[i,j]<1:
            x_specise.append(df_specise.loc[i,j])
            y_spearman.append(df_Sp.loc[i,j])
X=sm.add_constant(x_specise)
model=sm.OLS(y_spearman,X).fit()
print(model.summary())
plt.figure()
# plt.scatter(x_specise,y_spearman,alpha=0.2)
sb.regplot(x_specise,y_spearman,fit_reg=True,x_jitter=0.4,y_jitter=0.05,
           scatter_kws={"alpha":1/3},color="orange")
plt.text(7,0.9,r"$r^2=0.254***$",fontsize=13)
plt.xlabel('Specises',fontsize=15)
plt.ylabel('Best_Spearman',fontsize=15)
plt.show()