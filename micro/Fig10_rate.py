# -*- coding: utf-8 -*-
# @Time:2021/5/10:23
# @File:Fig10_rate.py
import pandas as pd
import numpy as np

def ex_deal(df_Int, df_ex):
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            if int(df_Int.iloc[item, 0]) == int(df_ex.iloc[jtem, 1]):
                df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
    df_Int.drop([0, 19], inplace=True)
    return df_Int

def main():
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    ind = np.linspace(2008, 2020, 13)
    D = dict((k,0) for k in ["loop","chain","en"])
    df_cir = pd.read_excel(path + "Network/circle20.xls", sheet_name=str(int(2008)))
    df_cir = ex_deal(df_cir, df_ex)
    # for j in range(np.shape(df_cir)[0]):
    #     if df_cir.iloc[j, 1] > 0:
    #         D["loop"] += 1
    #     elif df_cir.iloc[j, 1] == 0:
    #         D["chain"] += 1
    #     else:
    #         D["en"] += 1
    sum = 0
    # for key in D.keys():
    #     sum += D[key]
    # print("2008sum", sum)
    # print("2008占比", (D["loop"]+D["chain"]) / sum)
    for year in ind:
        df_cir = pd.read_excel(path + "Network/circle20.xls", sheet_name=str(int(year)))
        df_cir = ex_deal(df_cir, df_ex)
        gb=df_cir.groupby(["氮素"])
        for g in gb:
            if int(g[0])==0:
                for i in range(np.shape(g[1])[0]):
                    if g[1].iloc[i,1]>0:
                        D["loop"]+=1
                    elif g[1].iloc[i,1]==0:
                        D["chain"]+=1
                    else:
                        D["en"] += 1
    for key in D.keys():
        sum += D[key]
    print("sum",sum)
    print("占比",D["loop"]+D["chain"])
    print("占比", D["loop"] / (D["loop"] + D["chain"]))
    print("占比", D["chain"] / (D["loop"] + D["chain"]))

main()