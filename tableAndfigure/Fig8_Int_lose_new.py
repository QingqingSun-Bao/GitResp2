"""非传递性与物种多样性的关系"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import statsmodels.api as sm
import seaborn as sb

'''各种处理下物种组配的时间变化规律图'''
path1 = 'C:/Users/97899/Desktop/N/Richness/rich_null.xls'
path = 'C:/Users/97899/Desktop/PU/Intransitivity/Tao_C.xls'
df_Int = pd.read_excel(path, sheet_name='Int')
# print(df_Int)

rich = ["new", "lose"]
'''Intransitivity and new / lose'''

for dtem in rich:
    df_rich = pd.read_excel(path1, sheet_name=dtem)
    ind = np.linspace(2009, 2019, 11).tolist()
    plt.figure()
    te = []
    Int = []
    for i in range(38):
        for j in ind:
            if -0.15 < df_Int.loc[i, j] < 1:
                te.append(df_rich.loc[i, j])
                Int.append(df_Int.loc[i, j])
    # x = sm.add_constant(Int)
    x = sm.add_constant(te)
    model = sm.OLS(Int, x)
    results = model.fit()
    print(results.summary())
    y_fitted = results.fittedvalues
    # plt.title("N=" + str(int(g[0])) + ',R2=' + str(R2[l - 1]))
    # plt.scatter(te, Int, c='b', s=50, marker='o')
    sb.regplot(te, Int, fit_reg=True, x_jitter=0.3, y_jitter=0.05, scatter_kws={"alpha": 1 / 3},
               color="red",marker="v")
    if dtem =="new":
        # plt.text(9,0.8,r"$r^2=0.013*$",fontsize=13)
        plt.xlabel("Specise_new", fontsize=13)
    if dtem =="lose":
        # plt.text(9, 0.9, r"$r^2<0.01$", fontsize=13)
        plt.xlabel("Specise_lose", fontsize=13)
    plt.ylabel('Intransitivity', fontsize=13)

    # plt.plot(x, y_fitted, 'r')
    # pd.DataFrame((x,Int)).T.to_excel('C:/Users/97899/Desktop/N/Int_Rich.xls')
    plt.ylim(0, 1.1)
    plt.show()
