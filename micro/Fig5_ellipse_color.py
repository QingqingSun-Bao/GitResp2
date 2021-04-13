import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.api as sm
from scipy.stats import pearsonr


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def all_bio_zhu(data, ex, zuhe, spec):
    bio = 0
    zhu = 0
    data = data[data["顺序"] == str(float(ex))]
    for zh in zuhe:
        dt1 = data[data["样地号"] == zh]
        bio_ = 0
        zhu_ = 0
        # 某块样地的总生物量
        for item in spec:
            bio_p = dt1[dt1["物种"] == item]["干重g"]
            bio_ = bio_ + float(bio_p)
            # 某块样地的总株丛数
            zhu_p = dt1[dt1["物种"] == item]["株丛数"].values
            for j in zhu_p:
                if j is not None:
                    if j==0:
                        continue
                    else:
                        zhu_ = zhu_ + float(j)
        bio = bio + bio_
        zhu = zhu + zhu_
    avg_bio = bio / len(zuhe)
    avg_zhu = zhu / len(zuhe)
    return avg_bio, avg_zhu


def main():
    path = "C:/Users/97899/Desktop/N/"
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    deal_ex = pd.read_excel(path + "实验处理_ex.xls")
    deal_ex.set_index(["顺序"], inplace=True)
    ring = {}
    chain = {}
    ring["bio"], ring["zhu"], ring["N"],  ring["M"], ring["F"],ring["N_"] = [], [], [], [], [],[]
    chain["bio"], chain["zhu"], chain["N"],  chain["M"], chain["F"],chain["N_"] = [], [], [], [], [],[]
    for year in range(2008, 2021):
        ring_chain = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(year))
        ring_chain.set_index(['Unnamed: 0'], inplace=True)
        bio_root = pd.read_sql(str(year), con=engine)
        print(year)
        for ex in range(1, 39):
            if ex==1 or ex==20:
                continue
            else:
                if ring_chain.loc[ex, 3] >= 0:
                    if ring_chain.loc[ex, 3] == 0:
                        bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                        chain["bio"].append(bio)
                        chain["zhu"].append(zhu)
                        chain["N"].append(deal_ex.loc[ex, "氮素"])
                        chain["M"].append(deal_ex.loc[ex, "刈割"])
                        chain["F"].append(deal_ex.loc[ex, "频率"])
                        if deal_ex.loc[ex, "氮素"]>=15:
                            chain["N_"].append(1)
                        else:
                            chain["N_"].append(0)
                    else:
                        bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                        ring["bio"].append(bio)
                        ring["zhu"].append(zhu)
                        ring["N"].append(deal_ex.loc[ex, "氮素"])
                        ring["M"].append(deal_ex.loc[ex, "刈割"])
                        ring["F"].append(deal_ex.loc[ex, "频率"])
                        if deal_ex.loc[ex, "氮素"]>=15:
                            ring["N_"].append(1)
                        else:
                            ring["N_"].append(0)

    ring_df=pd.DataFrame(ring)
    chain_df=pd.DataFrame(chain)
    # ring_df.to_excel(path +"tuoyuan.xls")
    # chain_df.to_excel(path+"tuoyuan_chain.xls")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6))

    extre_x = np.max(ring["bio"]) - np.min(ring["bio"])
    x1 = np.array([(i - np.min(ring["bio"])) / extre_x for i in ring["bio"]])
    extre_y = np.max(ring["zhu"]) - np.min(ring["zhu"])
    y1 = np.array([(i - np.min(ring["zhu"])) / extre_y for i in ring["zhu"]])
    mu0 = np.mean(x1)
    mu1 = np.mean(y1)

    extre_x2 = np.max(chain["bio"]) - np.min(chain["bio"])
    x2 = np.array([(i - np.min(chain["bio"])) / extre_x2 for i in chain["bio"]])
    extre_y2 = np.max(chain["zhu"]) - np.min(chain["zhu"])
    y2 = np.array([(i - np.min(chain["zhu"])) / extre_y2 for i in chain["zhu"]])
    mu2 = np.mean(x2)
    mu3 = np.mean(y2)

    ring_df["bio_std"]=x1.tolist()
    ring_df["zhu_std"]=y1.tolist()
    chain_df["bio_std"]=x2.tolist()
    chain_df["zhu_std"] = y2.tolist()

    '''相关性检测'''
    cof1 = pearsonr(x1, y1)
    cof2 = pearsonr(x2, y2)
    print("ring", cof1, "chain", cof2)

    N = [0, 1, 2, 3, 5, 10, 15, 20, 50]
    color = ["black","red",   "gold",
            "sienna","lawngreen", "green",
             "teal", "blue", "fuchsia"]

    """更改分类变量"""
    groupby_variable="N_"
    # N_,M,F
    gbc1 = ring_df.groupby(groupby_variable)
    color1N=["green","fuchsia"]
    colorF=[ "blue","orange"]
    colorM=["black","red"]
    color_index=color1N

    for index, g in enumerate(gbc1):
           ax1.scatter(g[1].loc[:, "bio_std"], g[1].loc[:, "zhu_std"], marker="o", color=color_index[index], facecolor="none"
                    , s=50)
    # ax1.scatter(x1, y1, s=0.5)

    # # confidence_ellipse(x, y, axs, n_std=1,
    #                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x1, y1, ax1, n_std=2,
                        edgecolor='blue', linestyle='--',)#facecolor="skyblue",
    # # confidence_ellipse(x, y, axs, n_std=3,
    #                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax1.scatter(mu0, mu1, c='blue', s=15)
    ax1.text(0.8, 0.8, r"$r^{2}=0.101$")
    # ax1.set_title('ICN')
    # ax1.set_xlabel("Average Biomass ",fontsize=20)
    ax1.set_ylabel("TCN",fontsize=17)
    # 画水平与垂直线
    ax1.axhline(y=0.5,ls="--",C="black")
    ax1.axvline(x=0.5, ls="--", C="black")

    """更改分类变量"""
    gbc2 = chain_df.groupby(groupby_variable)

    for index, g in enumerate(gbc2):
            ax2.scatter(g[1].loc[:, "bio_std"], g[1].loc[:, "zhu_std"], marker="o", color=color_index[index],facecolor="none"
                        , s=60)
            print(g[0])
    # N1=["N<15","N>=15"]
    # N1=["M=No","M=Yes"]
    N1=["F=0","F=Low","F=High"]
    ax2.legend(ncol=1, bbox_to_anchor=(1, 0.5), fontsize=13, labels=N1)
    # ax2.scatter(x2, y2, s=0.5, c="red")

    # confidence_ellipse(x, y, axs, n_std=1,
    #                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x2, y2, ax2, n_std=2,
                        edgecolor='red', linestyle='--')

    ax2.scatter(mu2, mu3, c='red', s=15)

    ax2.text(0.7, 0.8, r"$r^{2}=0.494***$")
    # ax2.set_title('TCN')
    ax2.set_xlabel("Average Biomass",fontsize=20)
    ax2.set_ylabel("ICN",fontsize=17)
    # ax2.set_ylabel("Numnber of RP",fontsize=20)
    # 画水平与垂直线
    ax2.axhline(y=0.5, ls="--", C="black")
    ax2.axvline(x=0.5, ls="--", C="black")



    plt.show()




main()
