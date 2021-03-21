'''环链对物种多样性的影响,直方图'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib


# 导入组合
def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''Intransitivity and experiment deal'''


def ex_deal(df_Int, df_ex, Nature=False):
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            if int(df_Int.iloc[item, 0]) == int(df_ex.iloc[jtem, 1]):
                df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]

    if Nature == False:
        df_Int.drop([0, 19], inplace=True)
    return df_Int


def main():
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    df_rich = pd.read_excel(path + "richness/rich_null21.xls", sheet_name="alpha")
    df_rich.set_index("Unnamed: 0")
    '''环链生物量的差异'''
    D = {}
    D["loop"], D["chain"] = [], []
    N_loop, N_chain, loop_err, chain_err, Mannwh = [], [], [], [], []
    for year in [2008]:
        df_cir = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(int(year)))
        df_cir = ex_deal(df_cir, df_ex, Nature=True)
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
        gb = ex_deal(df_cir, df_ex).groupby("氮素")
        D_N["loop"][year], D_N["chain"][year] = {}, {}
        for g in gb:
            D_N["loop"][year][g[0]], D_N["chain"][year][g[0]] = 0, 0
            ls_loop = []
            ls_chain = []
            for index, cir in zip(g[1]["顺序"], g[1][3]):
                if cir > 0:
                    ls_loop.append(df_rich.loc[index - 1, year])
                if cir == 0:
                    ls_chain.append(df_rich.loc[index - 1, year])
            mean_rich_loop = np.mean(ls_loop) if len(ls_loop) > 0 else 0
            mean_rich_chain = np.mean(ls_chain) if len(ls_chain) > 0 else 0
            D_N["loop"][year][g[0]] = mean_rich_loop
            D_N["chain"][year][g[0]] = mean_rich_chain
    df_loop, df_chain = pd.DataFrame(D_N["loop"]), pd.DataFrame(D_N["chain"])
    print(df_loop, df_chain)
    '''画不同氮素下的雷达图'''
    lables = np.linspace(2008, 2020, 13).astype(int)
    nAttr = 13
    N_label=df_loop.index
    print(N_label)
    fig = plt.figure(facecolor="white")
    cmap = matplotlib.cm.get_cmap('Spectral')  # 可以选要提取的cmap，如'Spectral'
    cmap(0.1)  # 0-1
    n,m=1,1
    for item in N_label:
        rich=np.array(df_loop.loc[item,:])
        angles = np.linspace(0, 2*np.pi, nAttr, endpoint=False)
        rich= np.concatenate((rich, [rich[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        plt.subplot(121, polar=True)
        plt.plot(angles,rich, 'o-',c=cmap(0.1*n),label=item) #, linewidth = 2
        plt.thetagrids(angles * 180 / np.pi, lables)
        plt.grid(True)
        n=n+1
    plt.title("Loop",fontdict={"size":15})
    for item in N_label:
        rich=np.array(df_chain.loc[item,:])
        angles = np.linspace(0, 2*np.pi, nAttr, endpoint=False)
        rich= np.concatenate((rich, [rich[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        plt.subplot(122, polar=True)
        plt.plot(angles,rich, 'o-',c=cmap(0.1*m),label=item) #, linewidth = 2
        plt.thetagrids(angles * 180 / np.pi, lables)
        plt.grid(True)
        m=m+1
    plt.title("Chain",fontdict={"size":15})
    plt.legend(ncol=1, bbox_to_anchor=(1.1, 0.7), fontsize=13)
    plt.show()

main()

date = np.array([7, 5, 6, 9, 8, 7,0,0,0,0,0,0,0])
# lables = np.array(['综合','KDA','发育','推进','生存','输出'])
# nAttr = 6
# date = np.array([7, 5, 6, 9, 8, 7])
# angles = np.linspace(0, 2*np.pi, nAttr, endpoint=False)
# date = np.concatenate((date, [date[0]]))
# angles = np.concatenate((angles, [angles[0]]))
# fig = plt.figure(facecolor="white")
# plt.subplot(111, polar=True)
# plt.plot(angles, date, 'bo-', color = 'g', linewidth = 2)
# plt.fill(angles, date, facecolor = 'g', alpha = 0.25)
# plt.thetagrids(angles*180/np.pi, lables)
# plt.figtext(0.52, 0.95, 'DOTA能力值雷达图', ha='center')
# plt.grid(True)
# plt.savefig('dota_radar.JPG')
# plt.show()