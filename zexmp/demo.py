# -*- coding: utf-8 -*-
# @Time:2021/3/2116:28
# @File:demo.py
# @Software:PyCharm
import matplotlib.pyplot as plt
import numpy as np
data=[1,2,3,4,5,6,7,8,9]
category_colors = plt.get_cmap("viridis")(np.linspace(0, 1, 9))
n=0
for i,color in zip(data,category_colors):
    plt.bar(n,data,color=color)
    n+=1
plt.show()