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
    bio = []
    zhu = []
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
        bio.append(bio_)
        zhu.append(zhu_)
    cof=pearsonr(bio,zhu)
    print(cof[0])
    avg_bio = np.sum(bio) / len(zuhe)
    avg_zhu = np.sum(zhu) / len(zuhe)
    return avg_bio, avg_zhu,cof[0]


def main():
    path = "C:/Users/97899/Desktop/N/"
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    ring = {}
    ring["bio"],ring["zhu"] ,ring["cof"]  = [],[],[]
    chain = {}
    chain["bio"], chain["zhu"],chain["cof"], = [],[],[]
    for year in range(2008, 2021):
        ring_chain = pd.read_excel(path + "Network/circle20.xls", sheet_name=str(year))
        ring_chain.set_index(['Unnamed: 0'], inplace=True)
        bio_root = pd.read_sql(str(year), con=engine)
        print(year)
        for ex in range(1, 39):
            if ring_chain.loc[ex, 3] >= 0:
                if ring_chain.loc[ex, 3] == 0:
                    bio, zhu,cof = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                    chain["bio"].append(bio)
                    chain["zhu"].append(zhu)
                    chain["cof"].append(cof)
                else:
                    bio, zhu,cof = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                    ring["bio"].append(bio)
                    ring["zhu"].append(zhu)
                    ring["cof"].append(cof)


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

    '''相关性检测'''
    cof1 = pearsonr(x1, y1)
    cof2 = pearsonr(x2, y2)
    RoC = [1] * len(x1) + [0] * len(x2)
    C = {"type": RoC, "cof": ring["cof"] + chain["cof"],"zhu":ring["zhu"] + chain["zhu"],
         "bio":ring["bio"] + chain["bio"]}
    print(C["cof"])
    data = pd.DataFrame(C)
    data["intercept"] = 1.0
    train_cols = data.columns[1:]
    logit_ = sm.Logit(data["type"], data[train_cols])
    result = logit_.fit()
    print(result.summary())

    '''执行逻辑回归'''


main()
