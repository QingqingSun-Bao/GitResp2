"""非传递性与物种多样性的关系（刈割）"""
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
l = 1
plt.figure()
for dtem in diversity:
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
    gb_int = df_Int.groupby("频率")
    gb_rich = df_rich.groupby("频率")
    ind = np.linspace(2009, 2020, 12).tolist()

    """average data Intransitivity and diversity"""
    avg_int = {}
    avg_rich = {}
    x = []
    D_rich = {}
    D_rich["rich"], D_rich["Int"], D_rich["N_"] = [], [], []

    for gi, gr in zip(gb_int, gb_rich):
        print(gi[0])
        avg_int[gi[0]] = []
        avg_rich[gr[0]] = []
        N_ = []
        N_2 = []
        M = []
        F = []
        # print(type(gi[0]))
        # if gi[0] == 0:
        #     continue  # 控制氮素
        for ytem in ind:
            A = [i for i in gi[1][ytem] if 1 >= i > -0.15]
            B = [r for r in gr[1][ytem] if r > -0.15]
            if len(A) == 0:
                continue
            else:
                avg_int[gi[0]].extend(A)  # 某年氮素的平均值
                avg_rich[gr[0]].extend(B)
                N_.extend(np.repeat(gi[0],len(A)))
                # avg_int[gi[0]].append(np.mean(A))  # 某年氮素的平均值
                # avg_rich[gr[0]].append(np.mean(B))
                # N_.append(gi[0])
                # if gi[0] < 10:
                #     N_2.append(0)
                # else:
                #     N_2.append(0)
        # x.extend(np.repeat(math.log(float(gi[0]), 10), len(avg_int[gi[0]])))
        D_rich["rich"].extend(avg_rich[gi[0]])
        D_rich["Int"].extend(avg_int[gi[0]])
        D_rich["N_"].extend(N_)
    ri_In = pd.DataFrame(D_rich)
    """非传递性与物种多样性"""
    plt.subplot(1, 3, l)

    gri = ri_In.groupby("N_")
    for index, g in enumerate(gri):
        X = sm.add_constant(g[1]["rich"])
        te = np.array(g[1]["Int"])
        # X = sm.add_constant(ri_In["rich"])
        # te = np.array(ri_In["Int"])
        model = sm.OLS(te, X).fit()
        colorN = ["black", "sienna", "gold", "lawngreen", "green", "teal", "blue", "fuchsia"]
        colorN2 = ["red", "black"]
        colorF = ["sienna", "blue", "orange"]
        colorM = ["blue", "sienna"]
        # "red",
        # N = [1, 2, 3, 5, 10, 15, 20, 50]
        N2 = ["N<15", "N>15"]
        M = ["M=No", "M=Yes"]
        F = ["F=0", "F=Low", "F=High"]
        if dtem == "alpha":
            if g[0] == 0:
                plt.text(4, 0.4, r"$r^2=0.000$", fontsize=13)  # N
            else:
                plt.text(4, 0.6, r"$r^2=0.035**$", fontsize=13)  # N

            # sb.regplot(ri_In["rich"], ri_In["Int"], fit_reg=True,color="red",scatter_kws={"alpha": 0})
            sb.regplot(g[1]["rich"], g[1]["Int"], fit_reg=True, color="red", scatter_kws={"alpha": 0})
            plt.scatter(g[1]["rich"], g[1]["Int"], marker="s", facecolor="None", color=colorF[index])
            plt.xlabel("alpha-diversity", fontsize=15)
            plt.ylabel("Intransivity level", fontsize=15)
        if dtem == "gamma":
            if g[0] == 0:
                plt.text(6, 0.4, r"$r^2=0.003$", fontsize=13)  # N
            else:
                plt.text(5, 0.6, r"$r^2=0.021*$", fontsize=13)  # N
            # plt.text(0.25, 0.05, r"$r^2=0.128***$",fontsize=13)
            sb.regplot(g[1]["rich"], g[1]["Int"], fit_reg=True, scatter_kws={"alpha": 0},
                       color="green")
            plt.scatter(g[1]["rich"], g[1]["Int"], marker="s", facecolor="None", color=colorF[index])
            plt.xlabel("gamma-diversity", fontsize=15)
            plt.ylabel("Intransivity level", fontsize=15)
        if dtem == "beta":
            if g[0] == 0:
                plt.text(0, 0.6, r"$r^2=0.008$", fontsize=13)  # N
            else:
                plt.text(0, 0.4, r"$r^2=0.000$", fontsize=13)
            # plt.text(0.25, 0.8, r"$r^2=0.004$",fontsize=13) # N
            # plt.text(0.25, 0.05, r"$r^2=0.098***$", fontsize=13)
            plt.scatter(g[1]["rich"], g[1]["Int"], marker="s", facecolor="None", color=colorF[index])
            plt.legend(ncol=1, bbox_to_anchor=(1.1, 0.5), fontsize=13, labels=F)
            sb.regplot(g[1]["rich"], g[1]["Int"], fit_reg=True, scatter_kws={"alpha": 0},
                       color="orange")

            plt.xlabel("beta-diversity", fontsize=15)
            plt.ylabel("Intransivity level", fontsize=15)
        print(model.summary())

    l = l + 1
plt.show()
    # colorM = ["blue", "sienna"]
    # colorN2 = ["red", "black"]
    # M = [ "M=No", "M=Yes"]
    # N2 = ["N<10", "N>=10"]
    # plt.figure()
    # for index, g in enumerate(gri):
    #     plt.scatter(g[1]["rich"], g[1]["Int"], marker="s", facecolor="None", color=colorN2[index])
    # plt.legend(ncol=1, bbox_to_anchor=(1.1, 0.5), fontsize=13, labels=N2)
    # plt.show()



# '''Intransitivity and alpha / gamma/ beta rich————all data'''

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
