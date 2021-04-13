import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def ex_deal(df_Int, df_ex):
    columns = ['顺序', '氮素', '频率', '刈割']
    df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])
    for item in range(df_Int.shape[0]):
        for jtem in range(df_ex.shape[0]):
            if int(df_Int.iloc[item, 0] + 1) == int(df_ex.iloc[jtem, 1]):
                df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
                df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
                df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
                df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
    df_Int.drop([0, 19], inplace=True)
    return df_Int

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontdict={"size":15})

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),  ha="right"
             ) # rotation_mode="anchor"rotation=-30,

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor",  left=False)
    # bottom=False,

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def main():
    path = "C:/Users/97899/Desktop/N/"
    df = pd.read_excel(path + "Network/Strong_index.xls")
    df_ex = pd.read_excel(path + "实验处理_ex.xls")
    df = ex_deal(df, df_ex)
    df.set_index(["顺序"], inplace=True)
    ind = np.linspace(2008, 2020, 13)
    har = []
    gb = df.groupby(["氮素"])
    for g in gb:
        g1 = g[1].drop(['Unnamed: 0', '氮素', '频率', '刈割'], axis=1)
        N = []
        # print(g[1])
        for year in ind:
            # print(year,g[0])
            A = [i for i in g1.loc[:, int(year)] if i > -0.15]
            if len(A) == 0:
                N.append(-0.15)
            else:
                # 若包含1/2的链，1/2的概率为nan
                if np.mean(A) == 0:
                    N.append(0)
                else:
                    N.append(np.mean(A))
            # else:
            #     # 只考虑链与环
            #     if len(A) < 2 and np.mean(A) == 0:
            #         # 若只包含一个链，3/4的概率为nan，判定为nan
            #         N.append(0)
            #     else:
            #         N.append(np.mean(A))


        har.append(N)
    # print(np.mat(har).T)
    harvest = np.array(np.mat(har).T)
    # print("harvest",harvest)

    year = list(ind)
    N = ["0", "1", "2", "3", "5", "10", "15", "20", "50"]

    fig, ax = plt.subplots()
    im ,cbar= heatmap(harvest,year,N,ax=ax,cmap="viridis",cbarlabel="Network structure complexity")
    # texts=annotate_heatmap(im,vars("{x:.1f}t"))
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(N)))
    ax.set_yticks(np.arange(len(year)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(N)
    ax.set_yticklabels([int(i) for i in year])
    ax.set_ylabel("Year",fontdict={"size":15})
    ax.set_xlabel("N addition rate"r"$(gN m^{-2}year^{-1})$", fontdict={"size": 15})


    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(),  ha="right",
             rotation_mode="anchor")  # ,rotation=45,

    # Loop over data dimensions and create text annotations.
    for i in range(len(year)):
        for j in range(len(N)):
            if harvest[i, j]==-0.15:
                text = ax.text(j, i, "%s" % "NaN",
                               ha="center", va="center", color="w")
            else:
                text = ax.text(j, i, "%.2f" % harvest[i, j],
                           ha="center", va="center", color="w")

    # ax.set_title("Changes in network structure")
    fig.tight_layout()
    plt.show()
    """计算均值"""
    # mean_harvest=[]
    # print(harvest)
    # for item in harvest:
    #     mean_harvest.append(np.mean(item))
    #     print(np.mean(item))
    # print("均值", mean_harvest)
    """计算小于0.1的个数"""
    # count=[]
    # for item in harvest:
    #     i=0
    #     for j in item:
    #         if j <=0.02:
    #             i+=1
    #     count.append(-i)
    # print(count)



main()
