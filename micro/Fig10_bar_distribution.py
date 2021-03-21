import matplotlib.pyplot as plt
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


def MF(g, N="氮素"):
    if N == "刈割":
        if g == 0.0:
            g = "nm"
        else:
            g = "m"
    if N == "频率":
        if g == 2.0:
            g = "l"
        else:
            g = "h"
    return g


def loop_chain_nan(year, gb, D, N="氮素"):
    if year == 2008:
        for g_ in gb:
            print(MF(g_[0], N))
            g = MF(g_[0], N)
            D["loop"][g], D["nan"][g], D["chain"][g] = 0, 0, 0
            for item in g_[1][3]:
                if item == 0:
                    D["chain"][g] += 1
                elif item == -0.15:
                    D["nan"][g] += 1
                else:
                    D["loop"][g] += 1
    else:
        for g_ in gb:
            g= MF(g_[0], N)
            for item in g_[1][3]:
                if item == 0:
                    D["chain"][g] += 1
                elif item == -0.15:
                    D["nan"][g] += 1
                else:
                    D["loop"][g] += 1
    return D


def main():
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    ind = np.linspace(2008, 2020, 13)
    D = {}
    D["loop"], D["chain"], D["nan"] = {}, {}, {}
    for year in ind:
        df_cir = pd.read_excel(path + "Network/circle20.xls", sheet_name=str(int(year)))
        df_cir = ex_deal(df_cir, df_ex)
        gb = df_cir.groupby("氮素")
        D = loop_chain_nan(year, gb, D)
        gm = df_cir.groupby("刈割")
        D = loop_chain_nan(year, gm, D, "刈割")
        gf = df_cir.groupby("频率")
        D = loop_chain_nan(year, gf, D, "频率")

    print(D)
    net_loop = []
    net_chain = []
    net_nan = []
    '''氮素'''
    for key in D["loop"].keys():
        sum_ = D["loop"][key] + D["chain"][key] + D["nan"][key]
        print(sum_)
        net_loop.append(D["loop"][key] / sum_)
        net_chain.append(D["chain"][key] / sum_)
        net_nan.append(D["nan"][key] / sum_)
    print("非竞争", net_nan, "链", net_chain, "环", net_loop)
    labels = ['N=0', 'N=1', 'N=2', 'N=3', 'N=5', 'N=10', 'N=15', 'N=20', 'N=50']
    width = 0.5  # the width of the bars: can also be len(x) sequence

    net = (np.array(net_loop) + np.array(net_chain)).tolist()
    print("竞争主导", net)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax1=plt.subplot(212)
    ax1.bar(labels, net_loop[:9], width, label='Loop',color="darkcyan")
    ax1.bar(labels, net_chain[:9], width, bottom=net_loop[:9], label='Chain',color="turquoise")
    ax1.bar(labels, net_nan[:9], width, bottom=net[:9], label='E-D',color="yellow")
    ax1.set_ylabel('Ratio', fontdict={"size": 20})
    ax1.set_xlabel('N addition rater'"$(gN m^{-2}year^{-1})$", fontdict={"size": 15})

    width2 = 0.4
    '''刈割'''
    label_2 = ['No-M', 'M']
    ax2 = plt.subplot(221)
    ax2.bar(label_2, net_loop[9:11], width2, label='Loop', color="darkcyan")
    ax2.bar(label_2, net_chain[9:11], width2, bottom=net_loop[9:11], label='Chain', color="turquoise")
    ax2.bar(label_2, net_nan[9:11], width2, bottom=net[9:11], label='E-D', color="yellow")
    ax2.set_ylabel('Ratio', fontdict={"size": 20})
    ax2.set_xlabel('Mowing', fontdict={"size": 15})







    '''频率'''
    label_3 = ['Low', 'High']
    ax3 = plt.subplot(222)
    ax3.bar(label_3, net_loop[11:13],  width2, label='Loop', color="darkcyan")
    ax3.bar(label_3, net_chain[11:13],  width2, bottom=net_loop[11:13], label='Chain', color="turquoise")
    ax3.bar(label_3, net_nan[11:13],  width2, bottom=net[11:13], label='E-D', color="yellow")
    ax3.set_ylabel('Ratio', fontdict={"size": 20})
    ax3.set_xlabel('Frequency', fontdict={"size": 15})
    ax3.legend(ncol=1, bbox_to_anchor=(1.2, 1), fontsize=13)



    plt.show()


main()
