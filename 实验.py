from matplotlib import pyplot as plt
import numpy as np
import time
from math import *

x1 = np.linspace(1, 20, 20)
y1 = np.array(np.sin(x1))
print(x1,y1)
plt.plot(x1,y1)
plt.show()


# plt.ion() #开启interactive mode 成功的关键函数
# plt.figure(1)
# t = [0]
# t_now = 0
# m = [sin(t_now)]
#
# for i in range(2000):
#     plt.clf() #清空画布上的所有内容
#     t_now = i*0.1
#     t.append(t_now)#模拟数据增量流入，保存历史数据
#     m.append(sin(t_now))#模拟数据增量流入，保存历史数据
#     plt.plot(t,m,'-r')
#     plt.draw()#注意此函数需要调用
#     time.sleep(0.01)








# import numpy as np
# from functools import reduce
# import warnings
#
# warnings.filterwarnings("ignore", category=Warning)
#
#
# def perm(data, begin, end):
#     if begin == end:  # 递归结束条件，当交换到最后一个元素的时候不需要交换，1的全排列还是1。
#         print('begin == end=', begin, data)  # 打印一次排列完成后的数组。
#     else:
#         j = begin
#         # print('j = begin')
#         for i in range(begin, end):  # 从begin到end全排列。
#             # print('j=',j,'i=',i,data)
#             data[i], data[j] = data[j], data[i]
#             # print('交换位置')
#             perm(data, begin + 1, end)  # 递归结束
#             # print('perm之后的data',j,begin,data)
#             data[i], data[j] = data[j], data[i]  # 递归完成后，交换回原来的位置。
#             # print('换回原来的位置')
#
#
# '''验证P矩阵'''
#
#
# def CproductP(C_Mat):
#     n = C_Mat.shape[0]
#     P = np.mat(np.zeros(shape=(n, n)))
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 C_Arr = []
#                 for item in range(n):
#                     C_Arr.append(C_Mat[i, item])
#                 P[i, i] = reduce(lambda x, y: x * y, C_Arr)
#                 # 矩阵某行的连乘积reduce()
#             else:
#                 C_Arr = np.array(C_Mat[j, :])
#                 pp = reduce(lambda x, y: x * y, np.delete(C_Arr, C_Mat[j, i]))
#                 pp = pow(pp, 1 / (n - 2))
#                 # 几何平均值
#                 Geome_series = [pow(pp, i) for i in range(n - 1)]
#                 P[i, j] = 1 / (n - 1) * C_Mat[i, j] * reduce(lambda l, k: l + k, Geome_series)
#                 # 近似式，累加和
#     return P
#
#
# def main():
#     arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     # arr = ['灰背老鹳草', '大针茅', '硬质早熟禾', '砂韭', '瓣蕊唐松草', '野韭', '细叶韭', '细叶鸢尾', '糙隐子草', '刺藜', '猪毛菜', '羽茅', '灰绿藜', '冰草', '羊草', '轴藜', '二裂委陵菜', '黄囊苔草', '皱叶沙参', '羊茅', '猪毛蒿', '洽草', '小花花旗竿', '木地肤']
#     # A = perm(arr, 0, len(arr))
#     A = np.array([[1, 0.8, 0.8], [0.2, 1, 0.8], [0.2, 0.2, 1]])
#     B = np.mat(A)
#     P = CproductP(B)
#     print(P)
#
#
# main()


'''找出有向图中的环'''

#
# def findcircle(G):
#     node_set = set()
#     r = len(G)
#     have_in_zero = True
#     while have_in_zero:
#         have_in_zero = False
#         for i in range(r):
#             for row in G:
#                 print(row[i])
#             if i not in node_set and not any([row[i] for row in G]):
#                 print([row[i] for row in G])
#                 node_set.add(i)
#                 print(node_set)
#                 G[i] = [0] * r
#                 have_in_zero = True
#                 break
#     return False if len(node_set) == r else True
#
#
# '''寻找有向图的环'''
#
# from copy import deepcopy as dc
#
# # 用集合去除重复路径
# ans = set()
#
#
# def dfs(graph, trace, start):
#     trace = dc(trace)  # 深拷贝，对不同起点，走过的路径不同
#
#     # 如果下一个点在trace中，则返回环
#     if start in trace:
#         index = trace.index(start)
#         tmp = [str(i) for i in trace[index:]]
#         ans.add(str(' '.join(tmp)))
#         # print(trace[index:])
#         return
#
#     trace.append(start)
#
#     # 深度优先递归递归
#     for i in graph[start]:
#         dfs(graph, trace, i)
#
#
#
#
#
# def main():
#     graph = {1: [2], 2: [3,5], 3: [1, 4, 5], 4: [1], 5: [3,2]}  # 包含大小环test图
#     A=dfs(graph, [], 1)
#     print('ans',A)
#     # G = [[0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1], [1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1],
#     #      [0, 0, 0, 0, 0, 0]]
#     # # G = [[0,0,0,1,0],[1,0,0,0,0],[0,0,0,1,1],[0,0,0,0,0],[0,1,0,0,0]]
#     # have_circle = findcircle(G)
#     # print(have_circle)
#
# main()
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [3, 0, 0, 3, 0]
# women_means = [25, 32, 34, 20, 25]
# print(type(men_means ))
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
# print(x)
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
#
# fig.tight_layout()
#
# plt.show()