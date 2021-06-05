# -*- coding: utf-8 -*-
# @Time:2021/4/211:18
# @File:Fig2_ven_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


path = "C:/Users/97899/Desktop/N/"
df_all_loop = pd.read_excel(path + "Network/loop_type.xls")
dic_={}
for item in range(np.shape(df_all_loop)[0]):
    set_loop = df_all_loop.iloc[item, 3:8]
    for i in range(5):
        if set_loop[i] >0:
            set_loop[i]=i+1
    if tuple(set_loop) in dic_.keys():
        dic_[tuple(set_loop)]+=1
    else:
        dic_[tuple(set_loop)]=1


dic_set={}
for key in dic_.keys():
    dic_set[key]=[]
    for item in range(np.shape(df_all_loop)[0]):
        set_loop = df_all_loop.iloc[item, 3:8]
        if key==tuple(set_loop):
            dic_set[key].append(tuple(df_all_loop.iloc[item, 1:3]))
print(dic_set)



"""画venn图"""
sub_set=[]
for item in dic_.keys():
     sub_set.append(set(item)-{0})
print(dic_)

v=venn3([set([4]),set([3,4,5]),set([2,4,5])],
        set_labels=(" "," "," "),
        alpha=0.6,#透明度
        set_colors=["y","r","b"])
v.get_patch_by_id('111').set_alpha(0.6)
v.get_patch_by_id('111').set_color('yellow')
v.get_label_by_id('111').set_text('Nest')
v.get_patch_by_id('010').set_alpha(0.7)
v.get_patch_by_id('010').set_color('red')
v.get_label_by_id('010').set_text('Independent')
v.get_patch_by_id('001').set_alpha(0.6)
v.get_label_by_id('001').set_text('Long')
v.get_patch_by_id('011').set_alpha(0.6)
v.get_patch_by_id('011').set_color('g')
v.get_label_by_id('011').set_text('Cross')
plt.title("Sample Venn diagram")

plt.show()
