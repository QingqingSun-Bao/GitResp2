import pandas as pd
import numpy as np
from numpy import *


def Load_dic(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


# 获得最优的C矩阵
def LoadDataSet(dic):
    C_mat = []
    for key in dic.keys():
        C_mat.append(dic[key][0])
    return C_mat

# 根据矩阵C计算物种的排名
# def get_Rank(C_mat,Spe_Assemb):



# def main():
#     Sp_path = 'C:/Users/97899/Desktop/N/N_2009/Spearman/2009-0.txt'
#     SP_dic = Load_dic(Sp_path)
#     for ex in range(1, 39):
#         path1 = 'C:/Users/97899/Desktop/N/N_2009/Spearman/2009' + '-' + str(float(ex)) + '.txt'
#         dic = Load_dic(path1)
#         C_mat, P_mat = LoadDataSet(dic)
#         # Tao_C.append(TransitivityInC(C_mat))
#         # Tao_PP.append(TransitivityInP(P_mat))


main()
