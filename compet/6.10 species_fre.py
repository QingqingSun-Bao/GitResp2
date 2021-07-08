# -*- coding: utf-8 -*-
# @Time:2021/6/1010:14
# @File:6.10 species_fre.py
"""提取环链网络中各物种出现的频率"""
import pandas as pd
import numpy as np
from numpy import *
from scipy.stats import ttest_rel


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''筛选物种组合(矩阵，组合列表，Spearman列表)'''


def Select_Zuhe(CP_dic, Ass_lst, Sp_lst):
    Ass = []
    max_spear = 0
    max_index = 0
    for item in range(len(Sp_lst)):
        if Sp_lst[item][0] > max_spear:
            # print(max_spear,i)
            max_spear = Sp_lst[item][0]
            max_index = item
    if max_spear >= 0.6:
        C_mat = CP_dic[max_index][0]
        Ass = Ass_lst[max_index]
    else:
        C_mat = np.zeros((3, 3))
        max_spear = -0.15
    # print(C_mat,max_spear)
    return C_mat, Ass


"""将物种的平均竞争系数写入字典"""


def get_SpeCoeffi(Ass, Dic):
    for a in Ass:
        if a in Dic.keys():
            Dic[a]+=1
        else:
            Dic[a]=1
    return Dic


if __name__ == "__main__":
    path0 = "C:/Users/97899/Desktop/N/N_year/"
    path = "C:/Users/97899/Desktop/N/"
    str_df = pd.read_excel(path + "Network/Strong_index.xls")
    str_df.set_index('Unnamed: 0')
    """获得物种系数"""
    SpecCoffe_Ringdic = dict((k, {}) for k in range(1,39))
    SpecCoffe_Chaindic = dict((k, {}) for k in range(1,39))
    for year in range(2008, 2021):
        print(year)
        # 获取物种组合
        path1 = path0 + "N_" + str(year) + '/Assemb/' + str(year) + '-' + str(0) + '.txt'
        Specise_set = LoadDict(path1)
        # 获取Spearman值
        path3 = path0 + "N_" + str(year) + '/Spearman/' + str(year) + '-' + str(0) + '.txt'
        Spear_set = LoadDict(path3)
        # 根据Spearman值筛选物种组合
        for ex in range(1, 39):
            ex = float(ex)
            # 环
            path2 = path0 + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
            CP_mat = LoadDict(path2)
            if str_df.loc[ex - 1, year] > 0:
                if year < 2016:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
                else:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
                SpecCoffe_Ringdic[ex] = get_SpeCoeffi(Ass, SpecCoffe_Ringdic[ex])

            if str_df.loc[ex - 1, year] == 0:
                if year < 2016:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[str(ex)], Spear_set[str(ex)])
                else:
                    C_Mat, Ass = Select_Zuhe(CP_mat, Specise_set[ex], Spear_set[ex])
                SpecCoffe_Chaindic[ex]= get_SpeCoeffi(Ass, SpecCoffe_Chaindic[ex])
    # print("链",SpecCoffe_Chaindic)
    # print("环",SpecCoffe_Ringdic)
    print("链",pd.DataFrame(SpecCoffe_Chaindic).T)
    print("环",pd.DataFrame(SpecCoffe_Ringdic).T)
    pd.DataFrame(SpecCoffe_Chaindic).T.to_excel(path+"Competition/Specfre_chain.xls")
    pd.DataFrame(SpecCoffe_Ringdic).T.to_excel(path + "Competition/Specfre_ring.xls")
    """物种在环链中的平均物种系数"""
    spec = ["大针茅", "羊草", "羽茅", "冰草", "黄囊苔草", "糙隐子草", "灰绿藜", "洽草",
            "猪毛蒿", "猪毛菜", "细叶韭", "轴藜", "小花花旗竿", "刺藜", "硬质早熟禾", "细叶鸢尾",
            "砂韭", "星毛委陵菜"]
    meanCoffe = {}
    ring = []
    chain = []
    # for key in spec:
    #     meanCoffe[key] = [mean(SpecCoffe_Ringdic[key])]
    #     ring.append(mean(SpecCoffe_Ringdic[key]))
    #     if key in SpecCoffe_Chaindic.keys():
    #         meanCoffe[key].append(mean(SpecCoffe_Chaindic[key]))
    #         chain.append(mean(SpecCoffe_Chaindic[key]))
    #     else:
    #         meanCoffe[key].append(0)
    #         chain.append(0)
    # print(pd.DataFrame(meanCoffe).T)
    # # pd.DataFrame(meanCoffe,index=["Ring","Chain"]).T.to_excel(path+"Competition/SpecCompe_min.xls")
    # # print(mean)
    # # 配对样本T检验
    # print(ttest_rel(ring, chain))