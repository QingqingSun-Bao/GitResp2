# -*- coding: utf-8 -*-
# @Time:2021/5/1310:45
# @File:Fig10_ Stability.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"""计算稳定性"""
def ex_deal(df_Int, df_ex):
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    print(df_Int)
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            print(df_Int.iloc[item, 0])
            # if int(df_Int.iloc[item, 0] + 1) == int(df_ex.iloc[jtem, 1]):
            if item == jtem:
                df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
    # df_Int.drop([0, 19], inplace=True)
    return df_Int


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    df_bio = pd.read_excel(path + "Biomass/bio_all.xls")
    # df_rich = pd.read_excel(path + "Richness/rich_all.xls", sheet_name="alpha")
    df_loop = pd.read_excel(path + "Network/loop_NODF.xls")
    df_ex = pd.read_excel(path + "实验处理_ex.xls")

    df_bio = df_bio.set_index(['Unnamed: 0'])
    # df_rich = df_rich.set_index(['Unnamed: 0'])

    bio_N = dict([(k, []) for k in [0, 1, 2, 3, 5, 10, 15, 20, 50]])
    bio_sd = dict([(k, 0) for k in [0, 1, 2, 3, 5, 10, 15, 20, 50]])

    # 频率
    # bio_N = dict([(k, []) for k in [0, 2, 12]])
    # bio_sd = dict([(k, 0) for k in [0, 2, 12]])

    # 刈割
    # bio_N = dict([(k, []) for k in [0, 1]])
    # bio_sd = dict([(k, 0) for k in [0, 1]])

    df_bio = ex_deal(df_bio, df_ex)
    gb = df_bio.groupby("氮素")
    for g in gb:
        bio_meanbyyear = []
        for year in range(2008, 2021):
            bio_meanbyyear.append(np.mean(g[1].loc[:, year]))
        bio_N[g[0]] = [m / np.std(bio_meanbyyear) for m in bio_meanbyyear]

    for key in bio_N.keys():
        if int(key)==0:
            x = np.repeat(float(0), 13)
            print(x,bio_N[key])
            plt.scatter(float(0), np.mean(bio_N[key]))
            plt.errorbar(float(0), np.mean(bio_N[key]),yerr=np.std(bio_N[key]))
        else:
            x = np.repeat(float(np.log10(key))+0.1, 13)
            # x = np.repeat(float(key) + 0.1, 13)
            plt.scatter(float(np.log10(key))+0.1, np.mean(bio_N[key]))
            plt.errorbar(float(np.log10(key))+0.1, np.mean(bio_N[key]), yerr=np.std(bio_N[key]))
    plt.ylabel("Biomass Stability ", fontdict={"size": 12})
    plt.xlabel("Log10 N addition rate"r"$(gNm^{-2}year^{-1})$", fontdict={"size": 12})
    ax = plt.gca()
    y_major_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(1, 6)
    # # N addition rate，Frequency，mowing
    plt.title("All Samples")
    plt.show()

    # for key in bio_N.keys():
    #     bio_N

    #
    # gb = df_loop.groupby("N")
    #
    # for g in gb:
    #     gb1 = g[1].groupby("year")
    #     bio_meanbyyear = []
    #     for g1 in gb1:
    #         year = int(g1[0])
    #         bio_ = []
    #         for item in range(np.shape(g1[1])[0]):
    #             df_ex = g1[1].iloc[item, :]
    #             ex = df_ex["ex"]
    #             bio_.append(df_rich.loc[ex-1, year])
    #         bio_meanbyyear.append(np.mean(bio_))
    #     bio_N[g[0]] = [ m / np.std(bio_meanbyyear) for m in bio_meanbyyear]

    # """取出自然年份"""
    # df_test=df_bio
    # bio_2008 = []
    # N0=df_loop[df_loop["year"]==2008]
    # for item in range(np.shape(N0)[0]):
    #     df_ex = N0.iloc[item, :]
    #     ex = df_ex["ex"]
    #     bio_2008.append(df_test.loc[ex - 1, 2008])
    # df_loop = df_loop[df_loop["year"] != 2008]
    # """按氮素分组"""
    # gb = df_loop.groupby("N")
    # for g in gb:
    #     gb1 = g[1].groupby("year")
    #     bio_meanbyyear = []
    #     for g1 in gb1:
    #         year = int(g1[0])
    #         bio_ = []
    #         for item in range(np.shape(g1[1])[0]):
    #             df_ex = g1[1].iloc[item, :]
    #             ex = df_ex["ex"]
    #             bio_.append(df_test.loc[ex - 1, year])
    #         bio_meanbyyear.append(np.mean(bio_))
    #     if float(g[0])==0.0:
    #         bio_meanbyyear.append(np.mean(bio_2008))
    #     bio_N[g[0]] = [m / np.std(bio_meanbyyear) for m in bio_meanbyyear]
    # bio_N[50]=[2.4]
    # for key in bio_N.keys():
    #     if int(key) == 0:
    #         x = np.repeat(float(0), len(bio_N[key]))
    #         # plt.scatter(x, bio_N[key]) # bio_N[key]
    #         plt.scatter(float(0), np.mean(bio_N[key]))
    #         plt.errorbar(float(0), np.mean(bio_N[key]), yerr=np.std(bio_N[key]))
    #     else:
    #         # x = np.repeat(float(np.log10(key)) + 0.1, len(bio_N[key]))  # len(bio_N[key])
    #         # plt.scatter(float(key))
    #         plt.scatter(float(np.log10(key)) + 0.1, np.mean(bio_N[key]))
    #         plt.errorbar(float(np.log10(key)) + 0.1, np.mean(bio_N[key]), yerr=np.std(bio_N[key]))
    # plt.ylabel("Biomass Stability in loop",fontdict={"size":12})
    # plt.xlabel("Log10 N addition rate"r"$(gNm^{-2}year^{-1})$",fontdict={"size":12})
    # ax = plt.gca()
    # y_major_locator = MultipleLocator(0.5)
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.ylim(1,6,0.5)
    # plt.title("loop Samples")
    # plt.show()
    # print(bio_N)
