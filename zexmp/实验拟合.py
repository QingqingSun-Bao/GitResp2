from scipy import stats
import numpy as np
from fitter import Fitter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
path='C:/Users/97899/Desktop/zuhe.xls'
D={}
data1=[]
data2=[]
data3=[]
for i in range(2008,2021):
    D[str(i)] = pd.read_excel(path,sheet_name=str(i))
    # print(D[str(i)].columns)

data1 = list(stats.norm.rvs(loc=0, scale=2, size=70000))
# data2 = list(stats.norm.rvs(loc=0, scale=20, size=30000))

x=np.linspace(2008,2020,13)

# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f = Fitter(data1, distributions=['norm', 'chi2'])
f.fit()
f.summary()
# f.hist() #绘制组数=bins的标准化直方图
f.plot_pdf(names=None, Nbest=3, lw=2) #绘制分布的概率密度函数
plt.show()
plt.title('Three species assembly')
# print(f.summary())
