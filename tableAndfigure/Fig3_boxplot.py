"""不同N下非传递性的箱线图"""
import numpy as np
from fitter import Fitter
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator

# 每年38个处理的非传递性
path = 'C:/Users/97899/Desktop/N/Intransitivity/Tao_C20.xls'
path1 = 'C:/Users/97899/Desktop/N/实验处理_ex.xls'

df_Int = pd.read_excel(path)
df_ex = pd.read_excel(path1)
columns = ['顺序', '氮素', '频率', '刈割']
df_Int = pd.concat([df_Int, pd.DataFrame(columns=columns)])

'''Intransitivity and experiment deal'''

for item in range(df_Int.shape[0]):
    for jtem in range(df_ex.shape[0]):
        if int(df_Int.iloc[item, 0]) + 1 == int(df_ex.iloc[jtem, 1]):
            df_Int.loc[item, '顺序'] = df_ex.iloc[jtem, 1]
            df_Int.loc[item, '氮素'] = df_ex.iloc[jtem, 2]
            df_Int.loc[item, '频率'] = df_ex.iloc[jtem, 3]
            df_Int.loc[item, '刈割'] = df_ex.iloc[jtem, 4]
df_Int.drop([0, 19], inplace=True)
"""画非传递性的箱线图"""
# plt.figure()

gb1 = df_Int.groupby("氮素")
gb2 = df_Int.groupby("频率")
gb3 = df_Int.groupby("刈割")
var_location = []

'''各个氮素下整体的箱线图'''
cate = []
year_location = []
ind = np.linspace(2008, 2020, 13).tolist()
R_pd = []
for g in gb1:
    te = []
    te_ = []
    for item in ind:
        te.extend(g[1][item])
        te_.extend([i for i in g[1][item]])
    #      if i !=-0.15
    year_location.append(te_)
    count = 1 - len([i for i in te if i == -0.15]) / len(te)
    print("%s浓度" % str(g[0]), "共有%d个竞争网络,%d个完全竞争网络,%d个非传递性网络,%d个环境主导"
          % (len([i for i in te if i != -0.15]),len([i for i in te if 0 <= i < 0.1]),
             len([i for i in te if i >=0.1]),len([i for i in te if i == -0.15])))
    count1 = len([i for i in te if 0 <= i < 0.1]) / len([i for i in te if i != -0.15])
    cate.append('N=%d(%.2f %%,%.2f %%)' % (int(g[0]), count * 100, count1 * 100))
    R_pd.append(te)
plt.figure()
# plt.title('all')
f = plt.boxplot(
    x=year_location,
    patch_artist=True,  # 自己设置颜色marjker
    labels=cate,  # 添加X轴标签
    whis=0.5,
    showmeans=True,
    flierprops={"marker": 'o', "markerfacecolor": "black", "color": "black", "markersize": 5},
    # 异常值的颜色属性，color:轮廓颜色，marker:标记形状，markerfacecolor:填充色
    medianprops={"linestyle": 'solid', "color": "black", "linewidth": 3},  # 中位线颜色属性，linestyle:线型，color:线的颜色
    meanline=True
)

# 改变四个箱体的填充颜色
colors = ['pink', 'lightgreen', 'yellow', 'lightblue', 'navajowhite', 'lime', 'white', 'violet', 'orange']
for patch, color in zip(f['boxes'], colors):
    patch.set_facecolor(color)

# 统一三张图的坐标
x_major_locator = MultipleLocator(9)
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.ylim(-0.15, 1.1)
plt.ylabel("Intransitivity level", fontdict={"size": 20})
plt.xlabel('N addition rate'r'$(g N m^{-2} year^{-2})$', fontdict={"size": 17})
plt.xticks(rotation=10)
plt.show()

'''频率下整体的箱线图'''
cate = []
year_location = []
ind = np.linspace(2009, 2019, 11).tolist()
R_pd = []
for g in gb2:
    te = []
    for item in ind:
        te.extend(g[1][item])
    year_location.append(te)
    count = 1 - len([i for i in te if i == -0.15]) / len(te)
    count1 = len([i for i in te if 0 <= i < 0.1]) / len([i for i in te if i != -0.15])
    if g[0] == 2:
        s = "Low"
    else:
        s = 'High'
    print("%s频率" % str(g[0]), "共有%d个竞争网络,%d个完全竞争网络,%d个非传递性网络,%d个环境主导"
          % (len([i for i in te if i != -0.15]), len([i for i in te if 0 <= i < 0.1]),
             len([i for i in te if i >= 0.1]), len([i for i in te if i == -0.15])))
    cate.append('Fre=%s(%.2f %%,%.2f %%)' % (s, count * 100, count1 * 100))
    R_pd.append(te)
plt.figure()
f = plt.boxplot(
    x=year_location,
    patch_artist=True,  # 自己设置颜色marjker
    labels=cate,  # 添加X轴标签
    whis=0.5,
    showmeans=True,
    flierprops={"marker": 'o', "markerfacecolor": "black", "color": "black", "markersize": 5},
    # 异常值的颜色属性，color:轮廓颜色，marker:标记形状，markerfacecolor:填充色
    medianprops={"linestyle": 'solid', "color": "black", "linewidth": 3},  # 中位线颜色属性，linestyle:线型，color:线的颜色
    meanline=True
)

# 改变四个箱体的填充颜色
colors = ['pink', 'lightgreen', 'yellow', 'lightblue', 'navajowhite', 'lime', 'white', 'violet', 'orange']
for patch, color in zip(f['boxes'], colors):
    patch.set_facecolor(color)

# 统一三张图的坐标
x_major_locator = MultipleLocator(9)
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.ylim(-0.15, 1.1)
plt.ylabel("Intransitivity level", fontdict={"size": 15})
plt.xlabel('Frequency', fontdict={"size": 15})
plt.show()

'''频率下整体的箱线图'''
cate = []
year_location = []
ind = np.linspace(2009, 2019, 11).tolist()
R_pd = []
for g in gb3:
    te = []
    for item in ind:
        te.extend(g[1][item])
    year_location.append(te)
    count = 1 - len([i for i in te if i == -0.15]) / len(te)
    count1 = len([i for i in te if 0 <= i < 0.1]) / len([i for i in te if i != -0.15])
    if g[0] == 0:
        s = "No"
    else:
        s = 'Yes'
    cate.append('M=%s(%.2f %%,%.2f %%)' % (s, count * 100, count1 * 100))
    print("%s刈割" % str(g[0]), "共有%d个竞争网络,%d个完全竞争网络,%d个非传递性网络,%d个环境主导"
          % (len([i for i in te if i != -0.15]), len([i for i in te if 0 <= i < 0.1]),
             len([i for i in te if i >= 0.1]), len([i for i in te if i == -0.15])))
    R_pd.append(te)
plt.figure()
f = plt.boxplot(
    x=year_location,
    patch_artist=True,  # 自己设置颜色marjker
    labels=cate,  # 添加X轴标签
    whis=0.5,
    showmeans=True,
    flierprops={"marker": 'o', "markerfacecolor": "black", "color": "black", "markersize": 5},
    # 异常值的颜色属性，color:轮廓颜色，marker:标记形状，markerfacecolor:填充色
    medianprops={"linestyle": 'solid', "color": "black", "linewidth": 3},  # 中位线颜色属性，linestyle:线型，color:线的颜色
    meanline=True
)

# 改变四个箱体的填充颜色
colors = ['yellow', 'lightblue', 'pink', 'lightgreen', 'navajowhite', 'lime', 'white', 'violet', 'orange']
for patch, color in zip(f['boxes'], colors):
    patch.set_facecolor(color)

# 统一三张图的坐标
x_major_locator = MultipleLocator(9)
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.ylim(-0.15, 1.1)
plt.ylabel("Intransitivity level", fontdict={"size": 15})
plt.xlabel('Mowing', fontdict={"size": 15})
plt.show()

'''08年自然状态下的箱线图'''
# all_location=df_Int.iloc[:,2].values.tolist()
# print(all_location)
# plt.figure()
# f = plt.boxplot(
#         x=all_location,
#         patch_artist=True,  # 自己设置颜色
#         labels=["Nature_2008"],
#         showmeans=True,
#         whis=0.5,
#         boxprops={"color": "black", "facecolor": 'pink'},  # 箱体的颜色属性，color:边框色，facecolor:填充色
#         flierprops={"marker": 'o', "markerfacecolor": "darkorange", "color": "black", "alpha": 0.8},
#         # 异常值的颜色属性，color:轮廓颜色，marker:标记形状，markerfacecolor:填充色
#         medianprops={"linestyle": "solid", "color": "black","linewidth":5} , # 中位线颜色属性，linestyle:线型，color:线的颜色
#         meanline=True
#     )
# x_major_locator = MultipleLocator(9)
# ax = plt.gca()
# ax.yaxis.set_major_locator(MultipleLocator(0.2))
# plt.ylim(0, 1)
# plt.show()
