import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Intransitivity and experiment deal'''


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


'''bar'''


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height))
        # xytext=(0, 3),  # 3 points vertical offset
        # ,textcoords="offset points"
        # )ha='center', va='bottom'


'''多图'''


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


'''将数据按照氮素分组'''


def gb_N(df):
    case = []
    gb = df.groupby('氮素')
    for g in gb:
        g1 = g[1].drop(['顺序', '氮素', '频率', '刈割'], axis=1)
        g1 = g1[~g1.loc[:, 3].isin([-0.15])]
        print(g[0],g[1])
        # 判断是否存在种间竞争
        if g1.empty:
            case.append([-0.15])
            print("-0.15",g[1])
        else:
            case1 = []
            # 计算不同环类型的数量
            for item in g1.columns:
                A = [i for i in g1.loc[:, item]]
                # # 若A的长度小于2且
                # if len(A) < 2 and np.mean(A)==0:
                #     case1.append(0)
                #     break
                # else:
                case1.append(round(np.mean(A), 3))
            case.append(case1)
    return case


def main():
    path = "C:/Users/97899/Desktop/N/"
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    cases = {}
    cases_2 = []
    for year in range(2008, 2021):
        print(year)
        cases[year] = []
        df_cir = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(year))
        # 匹配相应的实验处理
        df_cir = ex_deal(df_cir, df_ex)
        df_cir.set_index(['Unnamed: 0'], inplace=True)
        # 将数据按氮素分组
        cases[year].append(gb_N(df_cir))
        cases_2.extend(gb_N(df_cir))
    # define a list of cases to plot
    # define the figure size and grid layout properties
    m = 0
    xx = [0, 1, 2, 3, 5, 10, 15, 20, 50]
    yy = list(np.linspace(2008, 2020, 13))
    l = 1
    for index, case in enumerate(cases_2):
        case1 = case + [0] * (7 - len(case))
        plt.subplot(13, 9, l)
        labels = list(np.linspace(3, 9, 7))
        plt.xticks(())
        plt.yticks(())
        plt.ylim(0, 15)
        if len(case) == 1:
            plt.bar(labels, [0 for item in range(len(labels))])
            plt.text(2.7, 0, "NaN")
        if len(case) > 1:
            if np.mean(case) == 0:
                plt.bar(labels, [0 for item in range(len(labels))])
                plt.text(3, 0, "0", fontdict={"size": 15})
            else:
                plt.bar(labels, case1)
        # 添加标题
        if l % 9 == 0:
            plt.twinx()  # 双Y轴
            plt.yticks(())
            plt.ylabel(str(int(yy[m])), fontdict={"size": 8})
            m = m + 1
        if (l-1)%9==0:
            plt.yticks((0, 5,10))
        if l <= 9:
            plt.title("N=%d" % xx[l - 1], fontdict={"size": 20})
        if l == 55:
            plt.ylabel("Circle Number", fontdict={"size": 20})
        if l > 108:
            plt.xticks((3, 6, 9))
        if l == 113:
            plt.xlabel("Circle Type", fontdict={"size": 20})

        l = l + 1

    plt.show()


main()
