# -*- coding: utf-8 -*-
# @Time:2021/3/2415:24
# @File:EX_Deal.py
import pandas as pd


'''Intransitivity and experiment deal'''
def ex_deal(df_Int):
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    df_ex = pd.read_excel("C:/Users/97899/Desktop/N/实验处理_ex.xls")
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            if int(df_Int.iloc[item, 0]) == int(df_ex.iloc[jtem, 1]):
                df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
    df_Int.drop([0, 19], inplace=True)
    df_Int.drop([2008.0], axis=1, inplace=True)
    return df_Int

if __name__=='__main__':
    path="C:/Users/97899/Desktop/N/Double/"
    df_ya = pd.read_excel(path + "yaAndhu/ya_hu_biomass.xls", sheet_name="ya_no_nested")
    print(ex_deal(df_ya))


