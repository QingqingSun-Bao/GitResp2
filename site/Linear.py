import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
'''自然状态下非传递性与环境的关系'''

def Linear(x, y):
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    y_fitted = results.fittedvalues
    plt.figure()
    plt.scatter(x, y, c='b', s=50, marker='o')
    plt.plot(x, y_fitted, 'r')
    """降雨量与温度"""
    # plt.text(max(x) - 50, max(y), "r2=0.655***", fontsize=15)
    # plt.xlabel('Rainfall(growing seasons)')
    # plt.text(max(x) - 60, max(y), "r2=0.610***", fontsize=15)
    # plt.xlabel('Rainfall')
    # plt.text(min(x), max(y), "r2=0.647***", fontsize=15)
    # plt.xlabel('Temperature(growing seasons)')
    # plt.text(min(x), max(y), "r2=0.441***", fontsize=15)
    # plt.xlabel('Temperature')
    """丰富度"""
    plt.text(min(x), max(y), "r2=0.663***", fontsize=15)
    plt.xlabel('Richness')
    plt.ylabel('Intransitivity')
    plt.show()


def main():
    path1 = 'C:/Users/97899/Desktop/N/Intransitivity/Tao_C6.xls'
    path2 = 'C:/Users/97899/Desktop/N/Richness/rich_site.xls'
    df_int = pd.read_excel(path1,sheet_name='Int')
    df_rain = pd.read_excel(path2)
    print(df_int)
    print(df_rain)
    x = df_rain.iloc[:, 1]
    y = df_int.iloc[:, 1]
    Linear(x, y)


main()
