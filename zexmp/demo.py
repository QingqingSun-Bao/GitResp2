# -*- coding: utf-8 -*-
# @Time:2021/3/2116:28
# @File:demo.py
# @Software:PyCharm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
N = 5
menMeans = (20, 35, 30, 35, -27)
womenMeans = (25, 32, 34, 20, -25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


plt.bar(ind, menMeans, width, yerr=menStd, label='Men')
plt.bar(ind, womenMeans, width,
            bottom=menMeans, yerr=womenStd, label='Women')
plt.errorbar(ind, womenMeans, yerr=womenStd,capsize=7,fmt="none")


# plt.plthline(0, color='grey', linewidth=0.8)
# plt.sylabel('Scores')
# plt.title('Scores by group and gender')
# plt.set_xticks(ind)
# plt.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.legend()

# # Label with label_type 'center' instead of the default 'edge'
# plt.bar_label(p1, label_type='center')
# plt.bar_label(p2, label_type='center')
# plt.bar_label(p2)

plt.show()