# coding: utf-8
import pandas as pd
from numpy import *
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

'''导入数据'''


def LoadDataSet(CP_dic, Ass_lit):
    print(Ass_lit)
    Ass = []
    max_spear = 0
    max_index = 0
    i=0
    for item in Ass_lit:
        # print(item, len(Sp_lst))

        if item[1] > max_spear:
            # print(max_spear,i)
            max_spear = item[1]
            max_index = i
        i = i +1
    if max_spear >= 0.6:
        # print(CP_dic)
        C_mat = CP_dic[max_index][0]
        P_mat = CP_dic[max_index][1]
        Ass = Ass_lit[max_index][0]
    else:
        C_mat = np.zeros((3, 3))
        P_mat = np.zeros((3, 3))
        max_spear = -0.15
    return C_mat, P_mat, Ass, max_spear


def Loaddic(path):
    fr = open(path, encoding='UTF-8')
    # 'unicode_escape'
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''计算C矩阵的传递性'''


def TransitivityInC(C_mat):
    tao_c = []
    if np.all(C_mat == 0):
        return -0.15
    # print(pd.DataFrame(C_mat))
    else:
        M = C_mat.shape[0]
        sum_N = 0
        for i in range(M):
            count = 0
            for j in range(i + 1, M):
                if C_mat[i, j] < C_mat[j, i]:
                    count = count + 1
            sum_N = sum_N + count
        Tao_C = (2 * sum_N) / (M * (M - 1))
        tao_c.append(1 - Tao_C)
    return mean(tao_c)


'''计算P矩阵的传递性'''


def TransitivityInP(P_mat):
    tao_P = []
    if np.all(P_mat == 0):
        return -0.15
    else:
        M = P_mat.shape[0]
        P_tiu = pd.DataFrame(np.triu(P_mat, 1))
        sum_N = 0
        for item in range(M):
            cou = 0
            for i in range(M):
                for j in range(i + 1, M):
                    if P_tiu.iloc[i, item] < P_tiu.iloc[j, item]:
                        cou = cou + 1
            sum_N = sum_N + cou
        Tao_P = 1 - 2 * sum_N / (M * (M - 1) * (M - 2))
        tao_P.append(1 - Tao_P)
    return Tao_P


"""保存文本"""


def savedict(datadict):
    file = open("C:/Users/97899/Desktop/PU/Zuhe/Zuhe.txt", "w")
    file.write(str(datadict))
    file.close()


def main():
    Intra_dic = {}
    Sp_values = {}
    Zuhe = {}
    numb_spe = {}
    Tao_P = {}
    for year in range(2008, 2020):
        path3 = 'C:/Users/97899/Desktop/PU/PU_' + str(year) + '/Assemb/' + str(year) + '-0.txt'
        Ass_Spear_dic = Loaddic(path3)
        Intra_dic[year] = []
        Tao_C = []
        Tao_PP = []
        Sp_best = []
        ass_len_best = []
        Ass_best = {}
        Intra_dic['ex'] = np.linspace(1, 38, 38).tolist()
        Sp_values['ex'] = np.linspace(1, 38, 38).tolist()
        print(year)
        for ex in range(1, 39):
            path1 = 'C:/Users/97899/Desktop/PU/PU_' + str(year) + '/CPmat/' + str(year) + '-' + str(float(ex)) + '.txt'
            CP_dic = Loaddic(path1)
            if year < 2016:
                C_mat, P_mat, Ass, Sp_max = LoadDataSet(CP_dic, Ass_Spear_dic[float(ex)])
            else:
                C_mat, P_mat, Ass, Sp_max = LoadDataSet(CP_dic, Ass_Spear_dic[float(ex)])
            Sp_best.append(Sp_max)
            ass_len_best.append(len(Ass))
            Tao_C.append(TransitivityInC(C_mat))
            Tao_PP.append(TransitivityInP(P_mat))
            Ass_best[ex] = Ass
        Intra_dic[year] = Tao_C
        Sp_values[year] = Sp_best
        Zuhe[year] = Ass_best
        numb_spe[year] = ass_len_best
        Tao_P[year] = Tao_P
    savedict(Zuhe)
    write = pd.ExcelWriter('C:/Users/97899/Desktop/PU/Intransitivity/Tao_C.xls')
    pd.DataFrame(Intra_dic).to_excel(write, sheet_name="Int")
    pd.DataFrame(Sp_values).to_excel(write, sheet_name="Spearman")
    pd.DataFrame(numb_spe).to_excel(write, sheet_name="zuhe")
    pd.DataFrame(Tao_PP).to_excel(write, sheet_name="Tao_P")
    write.save()
    write.close()


main()
