"""生物量在各年的分布"""
from fitter import Fitter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = 'SimHei'

'''分布拟合'''


def distribution(data1, year):
    f = Fitter(data1, distributions=['norm'])
    # 给定想要拟合的分布
    f.fit()
    f.summary()
    # summary提供分布的拟合优度，以及分布的直方图与概率密度曲线图
    # f.hist() #绘制组数=bins的标准化直方图
    # f.plot_pdf(names=None, Nbest=3, lw=2) #绘制分布的概率密度函数
    A = f.summary()
    print(A.loc['norm', 'aic'])
    plt.title('第%s年的生物量,AIC=%.2f,BIC=%.2f' % (str(year), A.loc['norm', 'aic'], A.loc['norm', 'bic']))
    plt.show()


'''多图'''


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


'''物种组配的分布规律'''
path = 'C:/Users/97899/Desktop/N/zuhe_all.xls'
D = {}
data = []
for i in range(2008, 2021):
    D[str(i)] = pd.read_excel(path, sheet_name=str(i))
    data.extend(D[str(i)].loc[:, 2])
    # print(D[str(i)].columns)
# distribution(data)

'''生物量分布'''
path1 = 'C:/Users/97899/Desktop/N/biomass.xls'
df_biomass = pd.read_excel(path1)
datas = []
l = 1
for i in range(2008, 2021):
    plt.subplot(5, 3, l)
    datas = list(df_biomass.loc[:, i])
    f = Fitter(datas, distributions=['norm'])
    f.fit()
    f.hist()
    f.plot_pdf()
    A = f.df_errors
    plt.title('%syear,AIC=%.2f,BIC=%.2f' % (str(i), A.loc['norm', 'aic'], A.loc['norm', 'bic']),
              fontdict={'weight': 'normal', 'size': 7})
    l = l + 1
plt.show()
