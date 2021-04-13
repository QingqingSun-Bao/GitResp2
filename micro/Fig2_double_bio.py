import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EX_Deal import ex_deal
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

'''Group'''
def gpby(df_Int):
    gb1 = df_Int.groupby("频率")
    gx = {}
    for g1 in gb1:
        gb2 = g1[1].groupby("刈割")
        gx[g1[0]] = {}
        for g2 in gb2:
            gb3 = g2[1].groupby("氮素")
            gx[g1[0]][g2[0]] = []
            for g3 in gb3:
                g0 = g3[1].drop(["顺序", "氮素", "刈割", "频率", "Unnamed: 0"], axis=1)
                mean = np.mean([i for i in g0.values[0] if i > 0])
                if np.isnan(mean):
                    gx[g1[0]][g2[0]].append(0)
                else:
                    gx[g1[0]][g2[0]].append(mean)
    print(gx)
    result = []
    for key1 in gx.keys():
        for key2 in gx[key1].keys():
            sum_ = sum(gx[key1][key2])
            result.append([i  for i in gx[key1][key2]])

    results = {
        'Low Frequency and no mowing': result[0],
        'Low Frequency and mowing': result[1],
        'High Frequency and no mowing': result[2],
        'High Frequency and mowing': result[3],
    }
    return results

'''多图'''
def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

'''Figure'''

def Fig(re1,re2):
    category_names = ['N=0', 'N=1', 'N=2', 'N=3', 'N=5', 'N=10', 'N=15', 'N=20', 'N=50']
    key=list(re1.keys())
    print("re1",re1)
    print("re2",re2)
    for k in key:
        width = 0.35       # the width of the bars: can also be len(x) sequence
        fig,ax=plt.subplots()
        labels = category_names
        "# all"
        # 黄囊苔草Carex korshinskyi、糙隐子草Cleistogenes squarrosa、羊草Leymus chinensis
        ax.bar(labels, re1[k], width,  label='Leymus chinensis',color="cornflowerblue")
        ax.bar(labels, re2[k], width,  bottom=re1[k],
               label='Carex korshinskyi',color="lightblue")
        ax.set_ylabel('Mean_biomass')
        # ax.set_ylabel('Mean_competition_coefficient')
        "# nested"
        # ax.bar(labels, re1[k], width,  label='Leymus chinensis',color="seagreen")
        # ax.bar(labels, re2[k], width,  bottom=re1[k],
        #        label='Carex korshinskyi',color="lightgreen")
        # ax.set_ylabel('Mean_biomass')
        # ax.set_ylabel('Mean_competition_coefficient')
        ax.set_title(k)
        # plt.ylim(0, 200)
        ax.legend(ncol=1, bbox_to_anchor=(0.8, 1),fontsize='small')
        plt.show()


def main():
    path = 'C:/Users/97899/Desktop/N/Double/'
    # path_D = {"huang/huang_compe.xls": ["coff_nested", "coff_"], "huang/huang_bio.xls": ["hu_nested", "hu_bio"]}
    df_ya = pd.read_excel(path + "yaAndhu/ya_hu_biomass.xls", sheet_name="ya_no_nested")
    df_hu = pd.read_excel(path + "yaAndhu/ya_hu_biomass.xls", sheet_name="hu_no_nested")
    gp_ya=gpby(ex_deal(df_ya))
    gp_hu=gpby(ex_deal(df_hu))
    Fig(gp_ya,gp_hu)

main()

