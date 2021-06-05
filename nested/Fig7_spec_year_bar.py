# -*- coding: utf-8 -*-
# @Time:2021/4/1913:06
# @File:Fig7_spec_year_bar.py
# -*- coding: utf-8 -*-
# @Time:2021/4/1620:52
# @File:Fig7_spec_bar.py
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""获得物种频率字典"""
# def get_spec_ferq(item,dic_spec):
#     if item[2] not in dic_spec.keys():
#         dic_spec[item[2]] = 1
#     else:
#         dic_spec[item[2]] += 1
#     return dic_spec

"""分年份获取占比"""


def get_ratio_byyear(lst):
    dic_spec = dict((k, {}) for k in ["nature", "pre", "last"])
    spec = ["大针茅", "羊草", "羽茅", '冰草', '黄囊苔草', '糙隐子草', '灰绿藜', '洽草', '猪毛蒿', '猪毛菜', '细叶韭',
            '轴藜', '小花花旗竿', '刺藜', '硬质早熟禾', '细叶鸢尾', '砂韭', '星毛委陵菜']
    # 初始化
    for key_time in dic_spec.keys():
        dic_spec[key_time] = dict((k, 0) for k in spec)
    n_ = dict((k, 0) for k in ["nature", "pre", "last"])
    ex=[1.0,2.0,3.0,20.0,21.0,22.0]
    #[1.0,2.0,3.0,20.0,21.0,22.0]
    for item in lst:
        if float(item[0]) == 2008.0 or float(item[1]) in ex:
            n_["nature"] += 1
            dic_spec["nature"][item[2]] += 1
        elif 2008.0 < float(item[0]) < 2014 and float(item[1]) not in ex:
            n_["pre"] += 1
            dic_spec["pre"][item[2]] += 1
        elif float(item[0]) > 2014 and float(item[1]) not in ex:
            n_["last"] += 1
            dic_spec["last"][item[2]] += 1
    for key_time in dic_spec.keys():
        for key_spec in dic_spec[key_time].keys():
            dic_spec[key_time][key_spec] = dic_spec[key_time][key_spec] / (n_[key_time])
    return dic_spec


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    dic_rank = LoadDict(path + "Attribute/groupbyrank.txt")

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    set_spec = set()
    dic_coloc = {'洽草': "cyan", '灰绿藜': "seagreen", '小花花旗竿': "red", '羽茅': "dodgerblue", '刺藜': "lawngreen",
                 '猪毛蒿': "lime", '硬质早熟禾': "green", '黄囊苔草': "yellow", '冰草': "deepskyblue", '细叶鸢尾': "orange",
                 '羊草': "royalblue", '轴藜': "darkorange", '砂韭': "pink", '猪毛菜': "purple", '细叶韭': "fuchsia",
                 '大针茅': "blue", '糙隐子草': "springgreen", '星毛委陵菜': "violet"}


    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    com=[">0.85","0.85~0.8","0.8~0.75","0.75~0.7","0.7~0.65","0.65~0.6",
          "0.6~0.55","0.55~0.5","0.5~0.45","0.45~0.4","0.4~0.35","<0.35"]
    # com = [">0.8", "0.8~0.7", "0.7~0.6", "0.6~0.5", "0.5~0.4", "0.4~0.3","0.3-0.2"]
    labels = ["Nature", "Preliminary", "Later"]
    # 遍历各个等级
    n_spec=[[],[],[]]
    for key_cpm in dic_rank.keys():
        print(key_cpm)
        get_raiobytime = get_ratio_byyear(dic_rank[key_cpm])
        print(get_raiobytime)
        bottom = [0,0,0]
        n = 1
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for spec in get_raiobytime["nature"].keys():
        #         lst=[get_raiobytime["nature"][spec],get_raiobytime["pre"][spec]
        #             ,get_raiobytime["last"][spec]]
        #         if n == 1:
        #             plt.bar(labels, lst, width=0.5, label=spec, color=dic_coloc[spec])
        #         else:
        #             plt.bar(labels, lst, bottom=bottom, width=0.5, label=spec, color=dic_coloc[spec])
        #         bottom =[bottom[i]+lst[i] for i in range(3)]
        #         n = n + 1
        # # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15), ncol=1, fontsize=15, frameon=False)
        # plt.title("Level of competition "+com[int(key_cpm)-1])
        # plt.show()
        n_nature=0
        n_pre=0
        n_last=0
        for spec in get_raiobytime["nature"].keys():
            if get_raiobytime["nature"][spec]>0:
                n_nature+=1
            if get_raiobytime["pre"][spec]>0:
                n_pre+=1
            if get_raiobytime["last"][spec] > 0:
                n_last += 1
        n_spec[0].append(n_nature)
        n_spec[1].append(n_pre)
        n_spec[2].append(n_last)

    print(ss.f_oneway(n_spec[2],n_spec[1]))


    # print(set_spec)
    #
    #
