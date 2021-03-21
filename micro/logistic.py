import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.api as sm
from scipy.stats import pearsonr
import copy
import math

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
                        zhu_ = zhu_ + float(j)

        bio = bio + bio_
        zhu = zhu + zhu_
    avg_bio = bio / len(zuhe)
    avg_zhu = zhu / len(zuhe)
    return avg_bio, avg_zhu


from matplotlib.colors import ListedColormap


# 定义函数，用于绘制决策边界
def plot_decision_boundary(model, x, y):
    color = ["#2b4750", "#dc2624", "b"]
    marker = ["o", "p", "x"]
    class_label = np.unique(y)
    cmap = ListedColormap(color[:len(class_label)])
    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1 = np.arrange(x1_min - 1, x1_max + 1, 0.02)
    x2 = np.arrange(x2_min - 1, x2_max + 1, 0.02)
    x1, x2 = np.meshgrid(x1, x2)
    Z = model.predict(np.array([x1.reval(), x2.reval()]).T).reshape(x1.shape)

    # 绘制使用颜色填充的等高线
    plt.contourf(x1, x2, Z, cmap=cmap, alpha=0.5)
    for i, class_ in enumerate(class_label):
        plt.scatter(x=x[y == class_, 0], y=x[y == class_, 1], cmap=cmap.colors[i], label=class_, marker=marker[i])
    plt.legend()
    plt.show()


def main():
    path = "C:/Users/97899/Desktop/N/"
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    deal_ex = pd.read_excel(path + "实验处理_ex.xls")
    deal_ex.set_index(["顺序"], inplace=True)
    temp = pd.read_excel(path + "Enveriment/weather_temp_20.xls")
    temp.set_index(['Unnamed: 0'], inplace=True)
    rain = pd.read_excel(path + "Enveriment/weather_rain_20.xls")
    rain.set_index(['Unnamed: 0'], inplace=True)
    # print(deal_ex)
    ring = {}
    chain = {}
    no_comp={}
    ring["bio"], ring["zhu"], ring["N"], ring["rain"], ring["temp"], ring["M"], ring["F"],ring["Year"] = [],[], [], [], [], [], [], []
    chain["bio"], chain["zhu"], chain["N"], chain["rain"], chain["temp"], chain["M"], chain[
        "F"],chain["Year"] = [], [], [], [], [], [], [],[]
    no_comp["bio"], no_comp["zhu"], no_comp["N"], no_comp["rain"], no_comp["temp"], no_comp["M"], no_comp[
        "F"], no_comp["Year"] = [], [], [], [], [], [], [], []
    ring_plot=0
    chain_plot=0
    for year in range(2008, 2021):
        ring_chain = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(year))
        ring_chain.set_index(['Unnamed: 0'], inplace=True)
        bio_root = pd.read_sql(str(year), con=engine)
        print(year)
        temp1 = temp.loc[year, 0]
        rain1 = rain.loc[year, "season"]
        for ex in range(1, 39):
            if ring_chain.loc[ex, 3] >= 0:
                # print(ex)
                if ring_chain.loc[ex, 3] == 0:
                    bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                    chain["bio"].append(bio)
                    chain["zhu"].append(zhu)
                    chain["N"].append(deal_ex.loc[ex, "氮素"])
                    chain["rain"].append(rain1)
                    chain["temp"].append(temp1)
                    chain["M"].append(deal_ex.loc[ex, "刈割"])
                    chain["F"].append(deal_ex.loc[ex, "频率"])
                    chain["Year"].append(year)
                    chain_plot += len(zuhe_plot[year][ex])
                else:
                    bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                    ring["bio"].append(bio)
                    ring["zhu"].append(zhu)
                    ring["N"].append(deal_ex.loc[ex, "氮素"])
                    ring["rain"].append(rain1)
                    ring["temp"].append(temp1)
                    ring["M"].append(deal_ex.loc[ex, "刈割"])
                    ring["F"].append(deal_ex.loc[ex, "频率"])
                    ring["Year"].append(year)
                    ring_plot = ring_plot+len(zuhe_plot[year][ex])
            else:
                bio, zhu = all_bio_zhu(bio_root, ex, zuhe_plot[year][ex], zuhe[year][ex])
                no_comp["bio"].append(bio)
                no_comp["zhu"].append(zhu)
                no_comp["N"].append(deal_ex.loc[ex, "氮素"])
                no_comp["rain"].append(rain1)
                no_comp["temp"].append(temp1)
                no_comp["M"].append(deal_ex.loc[ex, "刈割"])
                no_comp["F"].append(deal_ex.loc[ex, "频率"])
                no_comp["Year"].append(year)
    extre_x = np.max(ring["bio"]) - np.min(ring["bio"])
    x1 = np.array([(i - np.min(ring["bio"])) / extre_x for i in ring["bio"]])
    extre_y = np.max(ring["zhu"]) - np.min(ring["zhu"])
    y1 = np.array([(i - np.min(ring["zhu"])) / extre_y for i in ring["zhu"]])

    extre_x2 = np.max(chain["bio"]) - np.min(chain["bio"])
    x2 = np.array([(i - np.min(chain["bio"])) / extre_x2 for i in chain["bio"]])
    extre_y2 = np.max(chain["zhu"]) - np.min(chain["zhu"])
    y2 = np.array([(i - np.min(chain["zhu"])) / extre_y2 for i in chain["zhu"]])

    print("loop",ring_plot,"chain",chain_plot,"all",ring_plot+chain_plot)

    '''相关性检测'''
    cof1 = pearsonr(x1, y1)
    cof2 = pearsonr(x2, y2)
    print("ring", cof1, "chain", cof2)

    '''执行逻辑回归'''
    RoC = [1] * len(x1) + [0] * len(x2)
    C = {"type": RoC,"zhu":np.log(ring["zhu"] + chain["zhu"]),
                     "N":ring["N"] + chain["N"],
                      "M":ring["M"]+chain["M"], "rain": ring["rain"] + chain["rain"]
                      }
    # , "rain": ring["rain"] + chain["rain"]
    # "rain":ring["rain"] + chain["rain"],0.6859,"bio": list(x1) + list(x2),
    # "zhu": list(y1) + list(y2),0.67,"N": ring["N"] + chain["N"],0.66,"bio": list(x1) + list(x2),0.68
    # "zhu": ring["zhu"] + chain["zhu"],"N": ring["N"] + chain["N"], "bio": ring["bio"] + chain["bio"],
    data = pd.DataFrame(C)
    '''保存含有类别的数据'''
    # data.to_excel(path+"Classifier/data.xls")
    '''设置哑变量'''
    # dummy_ranks=pd.get_dummies(data["N"],prefix="N")
    # print(dummy_ranks.head())
    # keep_columns=["type"]
    # data=pd.DataFrame(data[keep_columns]).join(dummy_ranks.loc[:,"N_1.0":])
    # print(data.head())
    '''建立模型'''
    # 指定作为训练变量的列
    data["intercept"] = 1.0
    train_cols = data.columns[1:]
    print(data["type"], data[train_cols])
    logit_ = sm.Logit(data["type"], data[train_cols])
    result = logit_.fit()
    print(result.summary())

    combos = copy.deepcopy(data)
    predict_cols = combos.columns[1:]
    combos["intercept"] = 1.0
    combos["predict"] = result.predict(combos[predict_cols])
    print(combos["predict"])
    total=0
    total_ = 0
    hit = 0
    hit_=0
    for value in combos.values:
        predict = value[-1]
        type_ = int(value[0])
        if predict > 0.5:
            total += 1
            if type_ == 1:
                hit += 1
        if predict < 0.5:
            total_ += 1
            if type_ == 0:
                hit_ += 1
    print("Total: %d,Hit:%d,Precision1:%.2f,Precision1:%.2f" % (total, hit, 100.0 * hit / total,100.0 * hit_ / total_))



main()
