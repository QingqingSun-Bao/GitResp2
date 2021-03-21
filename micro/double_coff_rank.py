import pandas as pd
import numpy as np
from numpy import *

'''导入矩阵字典'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''导入数据'''


def LoadDataSet(CP_dic, Ass_lst, Sp_lst):
    # print( Ass_lst)
    Ass = []
    max_spear = 0
    max_index = 0
    for item in range(len(Sp_lst)):
        # print(item, len(Sp_lst))
        if Sp_lst[item][0] > max_spear:
            # print(max_spear,i)
            max_spear = Sp_lst[item][0]
            max_index = item
    if max_spear >= 0.6:
        # print(CP_dic)
        C_mat = CP_dic[max_index][0]
        P_mat = CP_dic[max_index][1]
        Ass = Ass_lst[max_index]
    else:
        C_mat = np.zeros((3, 3))
        P_mat = np.zeros((3, 3))
        max_spear = -0.15
    return C_mat, Ass


'''物种具体的竞争系数以及排名'''


def compe_coff(C_mat, ass, spec):
    D = {}
    row_index1 = 0
    rank1 = 0
    row_index2 = 0
    rank2 = 0
    for index, item in enumerate(ass):
        if item == spec[0]:
            row_index1 = index
        if item == spec[1]:
            row_index2 = index
     # 按照竞争等级排序
    C_mat = np.array(C_mat)
    for i in range(C_mat.shape[0]):
        D[ass[i]] = len([j for j in C_mat[i, :] if j > 0.5])
    D_sort = sorted(D.items(), key=lambda x: x[1], reverse=True)
    # print(D_sort)
    # print(ass)
    # print(row_index1,row_index2)
    for index,jtem in enumerate(D_sort):
        if jtem[0] ==spec[0]:
            # print(jtem[0])
            rank1 = index+1
        if jtem[0] == spec[1]:
            rank2 = index+1
    coff1 = C_mat[row_index1, row_index2]
    coff2 = 1-coff1
    return coff1, rank1,coff2, rank2


def main():
    path = "C:/Users/97899/Desktop/N/"
    two_nested_ex = LoadDict(path + "Zuhe/two_nested_ex.txt")
    two_ex = LoadDict(path + "Zuhe/two_ex.txt")
    zuhe=LoadDict(path + "Zuhe/Zuhe.txt")
    spec = ["黄囊苔草","羊草"]
    # spec = "羊草"
    hu_coff_nested = {};hu_sort_nested = {}
    hu_coff={};hu_sort={}
    ya_coff_nested = {};ya_sort_nested = {}
    ya_coff = {};ya_sort = {}
    for year in range(2008, 2020):
        path3 = path + 'N_' + str(year) + '/Assemb/' + str(year) + '-0.txt'
        Ass_dic = LoadDict(path3)
        path2 = path + 'N_' + str(year) + '/Spearman/' + str(year) + '-0.txt'
        Spear_dic = LoadDict(path2)
        hu_coff_nested[year] = {};hu_sort_nested[year] = {}
        hu_coff[year] = {};hu_sort[year] = {}
        ya_coff_nested[year] = {};ya_sort_nested[year] = {}
        ya_coff[year] = {};ya_sort[year] = {}
        print(year)
        for ex in range(1, 39):
            ex = float(ex)
            # nested
            if [year, ex] in two_nested_ex[("黄囊苔草","羊草")]:
                path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
                CP_dic = LoadDict(path2)
                if year < 2016:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[str(float(ex))], Spear_dic[str(float(ex))])
                    hu_coff_nested[year][ex], hu_sort_nested[year][ex],\
                    ya_coff_nested[year][ex], ya_sort_nested[year][ex] = compe_coff(C_mat, Ass, spec)

                else:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[float(ex)], Spear_dic[float(ex)])
                    hu_coff_nested[year][ex], hu_sort_nested[year][ex], \
                    ya_coff_nested[year][ex], ya_sort_nested[year][ex] = compe_coff(C_mat, Ass, spec)

            else:
                hu_coff_nested[year][ex], hu_sort_nested[year][ex], \
                ya_coff_nested[year][ex], ya_sort_nested[year][ex]=0,0,0,0
                # all
            if [year, ex] in two_ex[("黄囊苔草","羊草")]:
                path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
                CP_dic = LoadDict(path2)
                if year < 2016:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[str(float(ex))], Spear_dic[str(float(ex))])
                    hu_coff[year][ex],hu_sort[year][ex],ya_coff[year][ex],ya_sort[year][ex] = compe_coff(C_mat, Ass, spec)
                else:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[float(ex)], Spear_dic[float(ex)])
                    hu_coff[year][ex],hu_sort[year][ex],ya_coff[year][ex],ya_sort[year][ex] = compe_coff(C_mat, Ass, spec)
            else:
                hu_coff[year][ex],hu_sort[year][ex],ya_coff[year][ex],ya_sort[year][ex] = 0,0,0,0
    # print(pd.DataFrame(D_mat))
    write = pd.ExcelWriter(path + "yaAndhu/yahu_compe.xls")
    pd.DataFrame(hu_coff_nested).to_excel(write, sheet_name="hu_coff_nested")
    pd.DataFrame(hu_sort_nested).to_excel(write, sheet_name="hu_rank_nested")
    pd.DataFrame(hu_coff).to_excel(write, sheet_name="hu_coff_")
    pd.DataFrame(hu_sort).to_excel(write, sheet_name="hu_rank_")
    pd.DataFrame(ya_coff_nested).to_excel(write, sheet_name="ya_coff_nested")
    pd.DataFrame(ya_sort_nested).to_excel(write, sheet_name="ya_rank_nested")
    pd.DataFrame(ya_coff).to_excel(write, sheet_name="ya_coff_")
    pd.DataFrame(ya_sort).to_excel(write, sheet_name="ya_rank_")
    write.save()
    write.close()


main()
