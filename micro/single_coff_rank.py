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
    row_index = 0
    rank = 0
    # # 找到物种所在的下标
    # for index, item in enumerate(ass):
    #     if item == spec:
    #         row_index = index
    #         break
    # # 按照竞争等级排序
    # print(ass)
    # print(ass[row_index],row_index)
    C_mat = np.array(C_mat)
    for i in range(C_mat.shape[0]):
        D[ass[i]] = len([j for j in C_mat[i, :] if j > 0.5])
    D_sort = sorted(D.items(), key=lambda x: x[1], reverse=True)
    print(C_mat)
    print(C_mat[1,:])
    print(D)
    print(D_sort)
    for index,jtem in enumerate(D_sort):
        if jtem[0] == spec:
            rank = index+1
            print(rank)
            break
    coff = np.mean(C_mat[row_index, :])
    # print(ass[jtem[0]])
    rank=(rank/len(ass))*6
    # print("new",rank)
    return coff, rank


def main():
    path = "C:/Users/97899/Desktop/N/"
    single_ex = LoadDict(path + "Zuhe/single_ex.txt")
    zuhe=LoadDict(path + "Zuhe/Zuhe.txt")
    spec = "黄囊苔草"
    # spec = "羊草"
    hu_coff_nested = {};hu_sort_nested = {}
    hu_coff={};hu_sort={}
    hu_no_coff = {};hu_no_sort = {}

    for year in range(2008, 2020):
        path3 = path + 'N_' + str(year) + '/Assemb/' + str(year) + '-0.txt'
        Ass_dic = LoadDict(path3)
        path2 = path + 'N_' + str(year) + '/Spearman/' + str(year) + '-0.txt'
        Spear_dic = LoadDict(path2)
        hu_coff_nested[year] = {};hu_sort_nested[year] = {}
        hu_coff[year] = {};hu_sort[year] = {}
        hu_no_coff[year] = {};hu_no_sort[year] = {}
        print(year)
        for ex in range(1, 39):
            ex = float(ex)
            # nested
            if [year, ex] in single_ex[spec]:
                path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
                CP_dic = LoadDict(path2)
                if year < 2016:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[str(float(ex))], Spear_dic[str(float(ex))])
                    coff, rank = compe_coff(C_mat, Ass, spec)
                    hu_coff_nested[year][ex] = coff
                    hu_sort_nested[year][ex] = rank
                else:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[float(ex)], Spear_dic[float(ex)])
                    coff, rank = compe_coff(C_mat, Ass, spec)
                    hu_coff_nested[year][ex] = coff
                    hu_sort_nested[year][ex] = rank

            else:
                hu_coff_nested[year][ex] = 0
                hu_sort_nested[year][ex] = 0
            # all
            if spec in zuhe[year][ex]:
                path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
                CP_dic = LoadDict(path2)
                if year < 2016:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[str(float(ex))], Spear_dic[str(float(ex))])
                    coff, rank = compe_coff(C_mat, Ass, spec)
                    hu_coff[year][ex] = coff
                    hu_sort[year][ex] = rank
                else:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[float(ex)], Spear_dic[float(ex)])
                    coff, rank = compe_coff(C_mat, Ass, spec)
                    hu_coff[year][ex] = coff
                    hu_sort[year][ex] = rank

            else:
                hu_coff[year][ex] = 0
                hu_sort[year][ex] = 0
            # no nested
            if (spec in zuhe[year][ex])and([year, ex] not in single_ex[spec]):
                path2 = path + "N_" + str(year) + '/CPmat/' + str(year) + '-' + str(ex) + '.txt'
                CP_dic = LoadDict(path2)
                if year < 2016:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[str(float(ex))], Spear_dic[str(float(ex))])
                    coff, rank = compe_coff(C_mat, Ass, spec)
                    hu_no_coff[year][ex] = coff
                    hu_no_sort[year][ex] = rank
                else:
                    C_mat, Ass = LoadDataSet(CP_dic, Ass_dic[float(ex)], Spear_dic[float(ex)])
                    coff, rank = compe_coff(C_mat, Ass, spec)
                    hu_no_coff[year][ex] = coff
                    hu_no_sort[year][ex] = rank

            else:
                hu_no_coff[year][ex] = 0
                hu_no_sort[year][ex] = 0
    # print(pd.DataFrame(D_mat))
    write = pd.ExcelWriter(path + "huang/huang_compe.xls")
    pd.DataFrame(hu_coff_nested).to_excel(write, sheet_name="coff_nested")
    pd.DataFrame(hu_sort_nested).to_excel(write, sheet_name="rank_nested")
    pd.DataFrame(hu_coff).to_excel(write, sheet_name="coff_all")
    pd.DataFrame(hu_sort).to_excel(write, sheet_name="rank_all")
    pd.DataFrame(hu_no_coff).to_excel(write, sheet_name="coff_no_nested")
    pd.DataFrame(hu_no_sort).to_excel(write, sheet_name="rank_no_nested")
    write.save()
    write.close()


main()
