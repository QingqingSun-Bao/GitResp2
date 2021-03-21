'''环链对物种多样性的影响,直方图'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# 导入组合
def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''Intransitivity and experiment deal'''


def ex_deal(df_Int, df_ex,Nature=False):
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            if int(df_Int.iloc[item, 0]) == int(df_ex.iloc[jtem, 1]):
                df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]

    if Nature==False:
        df_Int.drop([0, 19], inplace=True)
    return df_Int


def main():
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    df_rich = pd.read_excel(path + "richness/rich_null21.xls", sheet_name="alpha")
    df_rich.set_index("Unnamed: 0")
    # print(df_rich)
    # zuhe=LoadDict(path+"Zuhe/Zuhe_20")
    # 引入组合计算奇数偶数
    '''环链生物链的差异'''
    D = {}
    D["loop"], D["chain"] = [], []
    N_loop, N_chain, loop_err, chain_err, Mannwh = [], [], [], [], []
    for year in [2008]:
        df_cir = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(int(year)))
        df_cir= ex_deal(df_cir, df_ex,Nature=True)
        for index, cir in zip(df_cir["顺序"], df_cir[3]):
            if cir > 0:
                D["loop"].append(df_rich.loc[index - 1, year])
            if cir == 0:
                D["chain"].append(df_rich.loc[index - 1, year])
    Nat = stats.mannwhitneyu(np.array(D["loop"]), np.array(D["chain"]), alternative='two-sided')
    '''每年的平均值'''
    N_loop.append(np.mean(D["loop"]))
    N_chain.append(np.mean(D["chain"]))
    Mannwh.append(Nat)

    '''各个氮素下环链多样性的差异'''
    D_N = {}
    D_N["loop"], D_N["chain"] = {}, {}
    ind = np.linspace(2008, 2020, 13)
    for year in ind:
        df_cir = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(int(year)))
        # df_cir=ex_deal(df_cir, df_ex)
        gb = ex_deal(df_cir, df_ex).groupby("氮素")
        if year == 2008:
            for g in gb:
                print(g[0], g[1])
                D_N["loop"][g[0]], D_N["chain"][g[0]] = [], []
                # print(df_rich)
                # print(D_N["loop"][g[0]].append([100001]))
                for index, cir in zip(g[1]["顺序"], g[1][3]):
                    if cir > 0:
                        D_N["loop"][g[0]].append(df_rich.loc[index - 1, year])
                    if cir == 0:
                        D_N["chain"][g[0]].append(df_rich.loc[index - 1, year])
                print(D_N["loop"][g[0]], D_N["chain"][g[0]])
        else:
            for g in gb:
                for index, cir in zip(g[1]["顺序"], g[1][3]):
                    if cir > 0:
                        D_N["loop"][g[0]].append(df_rich.loc[index - 1, year])
                    if cir == 0:
                        D_N["chain"][g[0]].append(df_rich.loc[index - 1, year])
    # stats.mannwhitneyu(np.array(D["loop"]),np.array(D["chain"]),alternative='two-sided')
    # print(stats.mannwhitneyu(np.array(D["loop"]),np.array(D["chain"]),alternative="two-sided"))
    for k_1 in D_N["loop"].keys():
        print(k_1, D_N["loop"][k_1])
        N_loop.append(np.mean(D_N["loop"][k_1]))
        N_chain.append(np.mean(D_N["chain"][k_1]))
        loop_err.append(np.std(D_N["loop"][k_1]))
        chain_err.append(np.std(D_N["loop"][k_1]))
        M_w = stats.mannwhitneyu(np.array(D_N["loop"][k_1]), np.array(D_N["chain"][k_1]), alternative="two-sided")
        Mannwh.append(M_w)
    print(D_N)
    '''画出不同氮素下的条形图'''
    X_data = ["Nature", "N=0", "N=1", "N=2", "N=3", "N=5", "N=10", "N=15", "N=20", "N=50"]
    bar_with = 0.4
    plt.barh(y=range(len(X_data)), width=N_loop, label="Loop", color="darkcyan", alpha=0.8, height=bar_with)
    plt.barh(y=np.arange(len(X_data)) + bar_with, width=N_chain, color="turquoise", label="Chain", alpha=0.8,
             height=bar_with)
    xing = ""
    for index, x in enumerate(N_chain):
        if Mannwh[index][1] < 0.05 and Mannwh[index][1] > 0.01:
            xing = "*"
        if Mannwh[index][1] < 0.01 and Mannwh[index][1] > 0.001:
            xing = "**"
        if Mannwh[index][1] < 0.001:
            xing = "***"
        plt.text(x + 0.3, index + bar_with / 2, "%s" % (str(Mannwh[index][0]) + xing), ha="center", va="bottom",
                 size=13)
        xing = ""
    plt.xlabel("Specises Richness", size=20)
    plt.ylabel('N addition rater'"$(gN m^{-2}year^{-1})$", size=20)
    plt.xticks(size=15)
    plt.yticks(np.arange(len(X_data)) + bar_with / 2, X_data, size=15)
    plt.legend()
    plt.show()


main()
