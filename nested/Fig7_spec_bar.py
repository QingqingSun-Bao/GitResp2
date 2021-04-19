# -*- coding: utf-8 -*-
# @Time:2021/4/1620:52
# @File:Fig7_spec_bar.py
import pandas as pd
import matplotlib.pyplot as plt


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic
def get_spec_ratio(lst):
    dic_spec={}
    n=len(lst)
    for item in lst:
        if item[2] not in dic_spec.keys():
            dic_spec[item[2]]=1
        else:
            dic_spec[item[2]]+=1
    for key in dic_spec.keys():
        dic_spec[key]=dic_spec[key]/n
    return dic_spec


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    dic_rank = LoadDict(path + "Attribute/groupbyrank.txt")
    labels=[1]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    set_spec=set()
    dic_coloc={'洽草':"cyan", '灰绿藜':"seagreen", '小花花旗竿':"red", '羽茅':"dodgerblue", '刺藜':"lawngreen",
               '猪毛蒿':"lime", '硬质早熟禾':"green", '黄囊苔草':"yellow", '冰草':"deepskyblue", '细叶鸢尾':"orange",
               '羊草':"royalblue", '轴藜':"darkorange", '砂韭':"pink", '猪毛菜':"purple", '细叶韭':"fuchsia",
                '大针茅':"blue",'糙隐子草':"springgreen", '星毛委陵菜':"violet"}
    spec=["大针茅","羊草","羽茅",'冰草','黄囊苔草','糙隐子草','灰绿藜','洽草','猪毛蒿','猪毛菜','细叶韭',
          '轴藜','小花花旗竿','刺藜','硬质早熟禾','细叶鸢尾','砂韭','星毛委陵菜']
    for key in dic_rank.keys():
        print(key)
        get_raio = get_spec_ratio(dic_rank[key])
        print(get_raio)
        bottom = 0
        # set_spec=set_spec|set(get_raio.keys())
        # print(set_spec)
        n = 1
        for key1 in spec:
            if key1 in get_raio.keys():
                if n==1:
                   plt.bar(labels,get_raio[key1],width=0.03,label=key1,color=dic_coloc[key1])
                else:
                   plt.bar(labels, get_raio[key1], bottom=bottom,width=0.03,label=key1,color=dic_coloc[key1])
                bottom+=get_raio[key1]
                n=n+1
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15), ncol=1, fontsize=15,frameon=False)
        plt.show()
    # print(set_spec)


