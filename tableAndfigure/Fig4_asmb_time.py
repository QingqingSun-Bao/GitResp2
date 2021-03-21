from scipy import stats
import numpy as np
from fitter import Fitter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
from matplotlib.pyplot import MultipleLocator

'''各种处理下物种组配的时间变化规律图'''

path = 'C:/Users/97899/Desktop/'
D = {}
lt=[1,2,3,4,5,6,7,8,9]
D_asb = {}
for j in lt:
    l=[]
    for i in range(2008, 2021):
        D[str(i)] = pd.read_excel(path+'zuhe.xls', sheet_name=str(i))
        l.append(D[str(i)].loc[:, j])
    D_asb[j]=pd.DataFrame(l).T
    D_asb[j].columns = [np.linspace(2008, 2020, 13)]
axs = plt.figure(figsize= (10, 8), constrained_layout=False).subplots(nrows=8, ncols=5)
x = np.linspace(2008, 2020, 13)
x_new = np.linspace(x.min(), x.max(), 300)
def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

# 获取实验处理
indexs=pd.read_excel(path+'实验处理_ex.xls')
indexs=np.array(indexs).tolist()
cases = np.array(D_asb[9]).tolist()
axs = trim_axs(axs, len(cases))
tendency34_lt=[8,9,10,11,15,16,17,18,19,27,28,29,30,35,36,37,38]
tendency56_lt=[6,8,9,10,11,12,14,15,16,17,18,19,16,26,27,28,29,30,33,35,36,37,38]
tendency7_lt=[1,5,6,8,9,10,11,12,13,14,15,16,17,18,19,16,26,27,28,29,30,33,34,35,36,37,38]
tendency8_lf=[23,24,25,31]
tendency9_lf=[24]
for ax,case,index in zip(axs,cases,indexs):
    y_smooth = make_interp_spline(x, case)(x_new)
    ax.set_title('N=%s,Fre=%s,Mow=%s' % (str(int(index[2])),str(int(index[3])),str(int(index[4]))),
                fontdict={'weight':'normal','size': 7})
    #,'Fre=%s' % str(index[3]),'Mowing=%s' % str(index[4]
    if int(index[1]) in tendency9_lf:
        ax.plot(x_new, y_smooth, 'o', ls='-', ms=0.5)
    else:
        ax.plot(x_new, y_smooth, 'r', ls='-', ms=0.5)
    # ax.plot(x_new, y_smooth, 'o', ls='-', ms=0.5)
plt.show()
plt.title('Three species assembly')
# print(f.summary())
