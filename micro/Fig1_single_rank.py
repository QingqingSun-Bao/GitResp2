import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
# 每年38个处理的非传递性
path = 'C:/Users/97899/Desktop/N/'
path_D = {"huang/huang_compe.xls": ["rank_nested","rank_no_nested"]}
df_ex = pd.read_excel(path + "实验处理_ex.xls")
title = ["Competition ranking about Carex korshinskyi in nested",
         "Competition ranking about Carex korshinskyi in no nested",]
columns = ['顺序', '氮素', '频率', '刈割']
m = -1
for key in path_D.keys():
    for sheet in path_D[key]:

        df_Int = pd.read_excel(path + key, sheet_name=sheet)
        df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
        m=m+1
        '''Intransitivity and experiment deal'''

        for item in range(df_Int.shape[0]):
            for jtem in range(df_ex.shape[0]):
                if int(df_Int.iloc[item, 0]) == int(df_ex.iloc[jtem, 1]):
                    df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                    df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                    df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                    df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
        df_Int.drop([0, 19], inplace=True)
        df_Int.drop([2008.0], axis=1, inplace=True)

        '''Group'''
        gb1 = df_Int.groupby("频率")
        gx = {}
        for g1 in gb1:
            gb2 = g1[1].groupby("刈割")
            gx[g1[0]] = {}
            for g2 in gb2:
                gb3 = g2[1].groupby("氮素")
                gx[g1[0]][g2[0]] = []
                for g3 in gb3:
                    print(g3[0])
                    g0 = g3[1].drop(["顺序", "氮素", "刈割", "频率", "Unnamed: 0"], axis=1)
                    # print(g0)
                    mean = np.mean([i for i in g0.values[0] if i > 0])
                    if np.isnan(mean):
                        gx[g1[0]][g2[0]].append(0)
                    else:
                        gx[g1[0]][g2[0]].append(mean)
                    # print(gx[g1[0]][g2[0]])
        '''Figure'''

        result = []
        for key1 in gx.keys():
            for key2 in gx[key1].keys():
                sum_ = sum(gx[key1][key2])
                result.append([i for i in gx[key1][key2]])
        category_names = ['N=0', 'N=1', 'N=2', 'N=3', 'N=5', 'N=10', 'N=15', 'N=20', 'N=50']

        results = {
            'LF-NM': result[0],
            'LF-M': result[1],
            'HF-NM': result[2],
            'HF-M': result[3],
        }


        def survey(results, category_names):
            """
            Parameters
            ----------
            results : dict
                A mapping from question labels to a list of answers per category.
                It is assumed all lists contain the same number of entries and that
                it matches the length of *category_names*.
            category_names : list of str
                The category labels.
            """
            labels = list(results.keys())
            data = np.array(list(results.values()))
            data_cum = data.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, data.shape[1]))

            fig, ax = plt.subplots(figsize=(9.2, 5))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())

            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
                xcenters = starts + widths / 2
                print(labels)

                r, g, b, _ = color
                text_color = 'black' if 0 < r * g * b else "red"
                for y, (x, c) in enumerate(zip(xcenters, widths)):
                    if c == 0:
                        continue
                    else:
                        ax.text(x, y, str("%.2f" % c), ha='center', va='center',
                                color=text_color,fontsize=15)
            ax.legend(ncol=1, bbox_to_anchor=(1.1, 0.5), fontsize=13)
            ax.set_title(title[m],fontsize=20)


            return fig, ax


        # plt.title("Biomass")
        survey(results, category_names)
        plt.show()
