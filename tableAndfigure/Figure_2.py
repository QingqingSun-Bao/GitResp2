import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''画图：各年环数的分布状况柱状图'''

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = 'SimHei'
# np.random.seed(19680801)
# df=pd.read_excel('C:/Users/97899/Desktop/N_2009/环数_2009.xls',sheet_name='2009')
# n_bins = 5
# # print(df.values)
# # x = np.random.randn(1000, 1)
# x = df.iloc[:,1:5].values
# print(x)
# print(np.max(x))
# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
# # colors = ['red', 'brown', 'lime','blue','yellow','green']
# # labels=['3物种环','4物种环','5物种环','6环','7','8']
# colors=['red', 'brown', 'lime','blue']
# labels=['3物种环','4物种环','5物种环','6物种环']
# ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=labels)
# ax0.legend(prop={'size': 8})
# ax0.set_title('2009年非传递性环数量的分布')
#
# ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
# ax1.set_title('stacked bar')
#
# ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
# ax2.set_title('stack step (unfilled)')
#
# # Make a multiple-histogram of data-sets with different length.
# x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
# ax3.hist(x_multi, n_bins, histtype='bar')
# ax3.set_title('different sample sizes')
#
# fig.tight_layout()
# plt.show()

'''各物种环在不同氮素下的分布'''

df=pd.read_excel('C:/Users/97899/Desktop/N_2009/环数_2009.xls',sheet_name='2009')

# 分组统计# 环数柱状图
fig, ((ax0,ax1), (ax2,ax3)) = plt.subplots(nrows=2, ncols=2)
# df=df[df['氮素']!=2]
# df=df[df['氮素']!=3]
# new_ticks = np.linspace(1, 7, 7)
# labels = ['N-0', 'N-1',  'N-5', 'N-10', 'N-15', '  N-20', 'N-50']
new_ticks = np.linspace(1, 9, 9)
labels = ['N-0', 'N-1', 'N-2','N-3', 'N-5', 'N-10', 'N-15', '  N-20', 'N-50']
width = 0.4
x = np.arange(len(labels))
y=np.arange(7)

# 3 物种环
df_3=df.iloc[:,[1,6]].groupby('氮素').mean()
print(df_3.iloc[:,0])
ax0.bar(x, df_3.iloc[:,0], width, label='3物种环',color='red')
ax0.set_xticks(x)
ax0.set_yticks(y)
ax0.set_xticklabels(labels)
ax0.set_title('3物种环在不同氮素下的平均数量')
ax0.set_xlabel('氮素浓度')
ax0.set_ylabel('物种环的平均数量')
# 4 物种环
df_4=df.iloc[:,[2,6]].groupby('氮素').mean()
ax1.bar(x, df_4.iloc[:,0], width, label='4物种环',)
ax1.set_xticks(x)
ax1.set_yticks(y)
ax1.set_xticklabels(labels)
ax1.set_title('4物种环在不同氮素下的平均数量')
ax1.set_xlabel('氮素浓度')
ax1.set_ylabel('物种环的平均数量')
#
# 5 物种环
df_5=df.iloc[:,[3,6]].groupby('氮素').mean()
ax2.bar(x, df_5.iloc[:,0], width, label='5物种环',color='lime')
ax2.set_xticks(x)
ax2.set_yticks(y)
ax2.set_xticklabels(labels)
ax2.set_title('5物种环在不同氮素下的平均数量')
ax2.set_xlabel('氮素浓度')
ax2.set_ylabel('物种环的平均数量')
# 6 /7物种环
df_7=df.iloc[:,[5,6]].groupby('氮素').mean()
ax3.bar(x, df_7.iloc[:,0], width, label='7物种环',color='yellow')
ax3.set_xticks(x)
ax3.set_yticks(y)
ax3.set_xticklabels(labels)
ax3.set_title('7物种环在不同氮素下的平均数量')
ax3.set_xlabel('氮素浓度')
ax3.set_ylabel('物种环的平均数量')


plt.show()
'''2008年自然状态下的平均物种环'''
# df=pd.read_excel('C:/Users/97899/Desktop/N_2009/环数_2009.xls',sheet_name='2008')
# fig,ax=plt.subplots()
# new_ticks = np.linspace(1, 6, 6)
# labels = ['3物种环', '4物种环', '5物种环', '6物种环','7物种环','8物种环']
# width = 0.4
# x = np.arange(len(labels))
# df_2008=df.iloc[:,1:7].mean()
# print(df_2008)
# ax.bar(x, df_2008, width,color='brown')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.set_title('自然状态下物种环的平均数量')
# ax.set_xlabel('物种环的类别')
# ax.set_ylabel('物种环的数量')
# plt.show()