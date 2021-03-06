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
    dic = eval(fr.read())  # ???str?????????dict
    fr.close()
    return dic


def all_bio_zhu(data, ex, zuhe, spec):
    bio = 0
    zhu = 0
    data = data[data["??????"] == str(float(ex))]
    for zh in zuhe:
        dt1 = data[data["?????????"] == zh]
        bio_ = 0
        zhu_ = 0
        # ???????????????????????????
        for item in spec:
            bio_p = dt1[dt1["??????"] == item]["??????g"]
            bio_ = bio_ + float(bio_p)
            # ???????????????????????????
            zhu_p = dt1[dt1["??????"] == item]["?????????"].values
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
    ring = {}
    ring["bio"] = []
    ring["zhu"] = []
    chain = {}
    chain["bio"] = []
    chain["zhu"] = []
    for year in range(2008, 2021):
        ring_chain = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(year))
        ring_chain.set_index(['Unnamed: 0'], inplace=True)
        bio_root = pd.read_sql(str(year), con=engine)
        print(year)
        for ex in range(1, 39):
            if ring_chain.loc[ex, 3] >= 0:
                if ring_chain.loc[ex, 3] == 0:
                    bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                    chain["bio"].append(bio)
                    chain["zhu"].append(zhu)
                else:
                    bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                    ring["bio"].append(bio)
                    ring["zhu"].append(zhu)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

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

    '''???????????????'''
    cof1 = pearsonr(x1, y1)
    cof2 = pearsonr(x2, y2)
    print("ring", cof1, "chain", cof2)

    ax1.scatter(x1, y1, s=0.5)

    # # confidence_ellipse(x, y, axs, n_std=1,
    #                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x1, y1, ax1, n_std=2, alpha=0.5,
                       facecolor="skyblue", edgecolor='blue', linestyle='--')
    # # confidence_ellipse(x, y, axs, n_std=3,
    #                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax1.scatter(mu0, mu1, c='blue', s=3)
    ax1.text(0.8, 0.8, "R=0.116")
    ax1.set_title('Loop')
    ax1.set_xlabel("Average Biomass ")
    ax1.set_ylabel("Number of Ramets")
    # ?????????????????????
    ax1.axhline(y=0.5,ls="--",C="yellow")
    ax1.axvline(x=0.5, ls="--", C="yellow")

    ax2.scatter(x2, y2, s=0.5, c="red")

    # confidence_ellipse(x, y, axs, n_std=1,
    #                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x2, y2, ax2, n_std=2, alpha=0.5,
                       facecolor="pink", edgecolor='red', linestyle='--')

    ax2.scatter(mu2, mu3, c='red', s=3)
    ax2.text(0.7, 0.8, "R=0.482***")
    ax2.set_title('Chained')
    ax2.set_xlabel("Average Biomass")
    ax2.set_ylabel("Number of Ramets")
    # ?????????????????????
    ax2.axhline(y=0.5, ls="--", C="yellow")
    ax2.axvline(x=0.5, ls="--", C="yellow")


    plt.show()


main()
