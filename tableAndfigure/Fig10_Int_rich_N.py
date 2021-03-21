"""非传递性与物种多样性的关系"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import statsmodels.api as sm
import math
import seaborn as sb

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

'''各种处理下物种组配的时间变化规律图'''
path2 = 'C:/Users/97899/Desktop/N/实验处理_ex.xls'
path1 = 'C:/Users/97899/Desktop/N/Richness/rich_null.xls'
path = 'C:/Users/97899/Desktop/N/Intransitivity/Tao_C20.xls'
diversity = ["alpha", "gamma", "beta"]
# diversity=["alpha"]
for dtem in diversity:
    print(dtem)
    df_Int = pd.read_excel(path, sheet_name='Int')
    df_rich = pd.read_excel(path1, sheet_name=dtem)
    df_ex = pd.read_excel(path2)
    """Intransitivity and ex"""
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    df_rich = pd.concat([df_rich, pd.DataFrame(columns=columns)])
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            if df_Int.iloc[item, 0] + 1 == df_ex.iloc[jtem, 1]:
                df_Int.loc[item, '顺序'] = df_rich.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_rich.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_rich.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_rich.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
    gb_int = df_Int.groupby("氮素")
    gb_rich = df_rich.groupby("氮素")
    ind = np.linspace(2009, 2020, 12).tolist()
    print(df_Int,df_rich,ind)
    """average data Intransitivity and diversity"""
    avg_int = {}
    avg_rich = {}
    x = []
    rich = []
    Int = []
    for gi, gr in zip(gb_int, gb_rich):
        avg_int[gi[0]] = []
        avg_rich[gr[0]] = []
        # print(type(gi[0]))
        if gi[0] == 0:
            continue
        for ytem in ind:
            A = [i for i in gi[1][ytem] if 1 >= i > -0.15]
            B = [r for r in gr[1][ytem] if r > -0.15]
            if len(A) == 0:
                continue
                # avg_int[gi[0]].append(0)
                # avg_rich[gr[0]].append(0)
            else:
                avg_int[gi[0]].append(np.mean(A))
                avg_rich[gr[0]].append(np.mean(B))
        x.extend(np.repeat(math.log(float(gi[0]), 10), len(avg_int[gi[0]])))
        rich.extend(avg_rich[gi[0]])
        Int.extend(avg_int[gi[0]])
    # print(len(x),len())



    """新物种植入与非传递性的关系"""
    """N素梯度与物种多样性"""
    #
    # X = sm.add_constant(x)
    # te = np.array(rich)
    # print(X,te)
    # model = sm.OLS(te, X).fit()
    # if dtem == "alpha":
    #     plt.ylabel(dtem + "-diversity", fontsize=15)
    #     # plt.text(0.5,2,r"$r^2=0.451***$",fontsize=13) # N
    #     plt.text(0.5, 4, r"$r^2=0.498***$", fontsize=13)
    #     plt.ylim(0,np.max(rich)+3)
    #     sb.regplot(x, rich, fit_reg=True,marker="s", color="red")
    #     # plt.scatter(x, rich,)
    # if dtem == "gamma":
    #     plt.ylabel(dtem + "-diversity", fontsize=15)
    #     # plt.text(0.5, 5, r"$r^2=0.433***$",fontsize=13) #N
    #     plt.text(0.5, 5, r"$r^2=0.448***$", fontsize=13)
    #     plt.ylim(0, np.max(rich) + 3)
    #     sb.regplot(x, rich, fit_reg=True,
    #                marker="s", color="green")
    # if dtem == "beta":
    #     plt.ylabel(dtem + "-diversity", fontsize=15)
    #     # plt.text(0.5, 0.1, r"$r^2=0.109***$",fontsize=13) #N
    #     plt.text(0.5, 0.25, r"$r^2=0.024$", fontsize=13)
    #     plt.ylim(0, 0.6)
    #     sb.regplot(x, rich, fit_reg=True,
    #                marker="s", color="orange")#x_jitter=0.05, y_jitter=0.05, scatter_kws={"alpha": 1 / 3},
    # print(model.summary())
    # plt.xlabel("log10 N addition rate"r"$gN m^{-2}year^{-1}$", fontsize=15)
    # plt.show()

    """非传递性与物种多样性"""
    X = sm.add_constant(rich)
    print(rich)
    te = np.array(Int)
    model = sm.OLS(te, X).fit()
    if dtem=="alpha":
        plt.xlabel(dtem + "-diversity", fontsize=15)
        plt.text(8,0.8,r"$r^2=0.088**$",fontsize=13) # N
        sb.regplot(rich, Int, fit_reg=True, marker="s",color="red",scatter_kws={"alpha": 1/3})
        # plt.scatter(rich, Int,marker="s")
    if dtem=="gamma":
        plt.xlabel(dtem + "-diversity", fontsize=15)
        plt.text(16, 0.9, r"$r^2=0.069*$",fontsize=13) #N
        # plt.text(0.25, 0.05, r"$r^2=0.128***$",fontsize=13)
        sb.regplot(rich, Int, fit_reg=True, scatter_kws={"alpha": 1 / 3},
               marker="s",color="green")
    if dtem=="beta":
        plt.xlabel(dtem + "-diversity", fontsize=15)
        plt.text(0.25, 0.8, r"$r^2=0.004$",fontsize=13) # N
        # plt.text(0.25, 0.05, r"$r^2=0.098***$", fontsize=13)
        sb.regplot(rich, Int, fit_reg=True,scatter_kws={"alpha": 1 / 3},
               marker="s",color="orange")
    print(model.summary())
    plt.ylabel("Intransivity level",fontsize=15)
    plt.show()

    '''Intransitivity and alpha / gamma/ beta rich————all data'''

    # plt.figure()
    # l = 1
    # te = []
    # Int = []
    # for gint, grich in zip(gb_int, gb_rich):
    #     int = []
    #     rich = []
    #     # print(gint,grich)
    #     for ytem in ind:
    #         A = [i for i in gint[1][ytem].values if -0.15 < i <= 1]
    #         int.extend(A)
    #         B = [j for j in grich[1][ytem].values if -0.15 < j]
    #         rich.extend(B)
    #         # print(item,len(A),len(B))
    #     X = sm.add_constant(rich)
    #     te = np.array(int)
    #     # print(len(int),len(rich))
    #     model = sm.OLS(te, X).fit()
    #     print("第%d个" % l)
    #     print(model.summary())
    #     y_fitted = model.fittedvalues
    #     plt.subplot(3, 3, l)
    #     if dtem == "alpha":
    #         R2 = ['0.084*', '0.003', '0', '0.003', '0.004', '0.034', '0.003', '0.112', '0.037']  # alpha多样性
    #         plt.text(np.max(rich) - 2, 0.8, r"$r^2=%s$" % R2[l - 1], fontsize=13)
    #     if dtem == "gamma":
    #         R2 = ["0.091*", "0.115*", "0", "0.017", "0.008", "0.171*", "0", "0.059", "0.110"]  # gamma多样性
    #         plt.text(np.max(rich) - 2, 0.8, r"$r^2=%s$" % R2[l - 1], fontsize=13)
    #     if dtem == "beta":
    #         print(dtem)
    #         R2 = ["0.049", "0.150*", "0", "0.117*", "0.053", "0.072", "0.003", "0.004", "0.098"]
    #         plt.text(0.25, 0.9, r"$r^2=%s$" % R2[l - 1], fontsize=13)
    #     plt.title("N=%s"r"$(g N m^{-2} year^{-2})$" % str(gint[0]), fontsize=11)
    #     plt.scatter(rich, int, c='g', s=50, marker='o')
    #     if l == 4:
    #         plt.ylabel("Intransitivity Level", fontsize=15)
    #     if l == 8:
    #         plt.xlabel(dtem + "-diversity", fontsize=15)
    #     plt.ylim(0, 1.0)
    #     plt.plot(rich, y_fitted, 'r')
    #     l = l + 1
    # plt.show()
