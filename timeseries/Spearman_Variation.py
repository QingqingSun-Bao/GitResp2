import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from functools import reduce
import time
import warnings
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=Warning)
rd = np.random.RandomState(23)
'''各个样地出现的总物种集合'''


def SetInTime(engine, year, S_con, ex):
    Observed_N = []
    sum_D = 0
    for i in year:
        data = pd.read_sql(str(i), con=engine)
        data_ex = data.loc[data['顺序'] == str(ex)]
        if len(S_con) == 0:
            S_con = set(data_ex.loc[:, '物种'])
        D = dict.fromkeys(S_con, int(0))
        # 将字典初始化，values全为0,键为S_con
        for j in S_con:
            set_lis = data_ex[data_ex['物种'] == j].index.tolist()
            for i in set_lis:
                D[j] = D[j] + float(data_ex.loc[i, '干重g'])
                # 干重g是第14列
            sum_D = sum([D[key] for key in D.keys()])
        Observed_N.append([D[key] / sum_D for key in S_con])
    Observed_N = pd.DataFrame(Observed_N, columns=S_con).T
    Observed_N.columns = [str(i) for i in year]

    return Observed_N, S_con


'''以生物量筛选物种筛选物种'''


def SelectInBiomass(engine, year, Observed_N, ex, number):
    D = dict.fromkeys(Observed_N.index.tolist(), 0)
    for key in Observed_N.index.tolist():
        D[key] = Observed_N.loc[key, '2008']
    # 转化成字典
    Select_Set = []
    Select_list = sorted(D.items(), key=lambda x: x[1], reverse=True)  # 按照value值对字典排序(从大到小)
    for i in range(0, number):
        Select_Set.append(Select_list[i][0])
    Select_ObN, A = SetInTime(engine, year, Select_Set, ex)

    return Select_Set, Select_ObN


'''以出现的频次筛选物种'''


def SelectInFreq(engine, year, ex, ):
    D = {}
    Select_Set = []
    for i in year:
        data = pd.read_sql(str(i), con=engine)
        data_ex = data.loc[data['顺序'] == str(ex)]
        Specise_Set = set(data_ex['物种'])
        for key in Specise_Set:
            if key in D.keys():
                D[key] = D[key] + 1
            else:
                D[key] = 1
    Freq_sort = sorted(D.items(), key=lambda x: x[1], reverse=True)
    for item in range(len(year) - 1):
        Select_Set.append(Freq_sort[item][0])
    Select_ObN = SetInTime(engine, year, Select_Set, ex)
    return Select_Set, Select_ObN


'''产生大量的竞争矩阵C'''


# 全局变量
def CompetitiveMatrices(Observed_N):
    n = Observed_N.shape[0]
    matrix_uniform = rd.uniform(0, 1, (n, n))
    Mat_Triu = np.mat(np.triu(matrix_uniform, 1))
    mat_tril = np.mat(np.mat(np.ones((n, n))) - np.mat(Mat_Triu.T))
    Mat_Tril = np.tril(mat_tril, 0)
    C_Mat = Mat_Tril + Mat_Triu

    return C_Mat


'''由矩阵C产生矩阵P'''


def CproductP(C_Mat):
    n = C_Mat.shape[0]
    P = np.mat(np.zeros(shape=(n, n)))
    for i in range(n):
        for j in range(n):
            if i == j:
                C_Arr = []
                for item in range(n):
                    C_Arr.append(C_Mat[i, item])
                P[i, i] = reduce(lambda x, y: x * y, C_Arr)
                # 矩阵某行的连乘积reduce()
            else:
                C_Arr = np.array(C_Mat[j, :])
                pp = reduce(lambda x, y: x * y, np.delete(C_Arr, i))
                pp = pow(pp, 1 / (n - 2))
                # 几何平均值
                Geome_series = [pow(pp, i) for i in range(n - 1)]
                P[i, j] = 1 / (n - 1) * C_Mat[i, j] * reduce(lambda l, k: l + k, Geome_series)
                # 近似式，累加和
    return P


'''利用Spearman寻找最优的矩阵C'''


def SpearCvsObserved(D, Observed_N1, Observed_N2):
    Max_Sp = 0
    best_key = 0
    best_P = []
    df_Observed = pd.DataFrame(Observed_N2)
    for key in D.keys():
        Pre_N = np.dot(D[key][1], Observed_N1)  # 做预测值,下一阶段的丰度数据
        df_preN = pd.DataFrame(Pre_N)
        df_preN.set_index(Observed_N1.index, inplace=True)
        spman = []
        P_val = []
        for j in range(Observed_N2.shape[1]):
            spman_temp = stats.mstats.spearmanr(df_preN.iloc[:, j], df_Observed.iloc[:, j], use_ties=True)
            spman.append(spman_temp[0])
            P_val.append(spman_temp[1])
        average_temp = np.mean(spman)
        if average_temp > Max_Sp:
            Max_Sp = average_temp
            best_key = key
            best_P = P_val

    return Max_Sp, best_key, best_P


def main():
    start = time.process_time()
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    Sp_value = []
    for i in range(2, 3):
        ex = float(i)
        year = np.linspace(2008, 2017, 10, dtype=int)  # 实验年份
        '''物种选择'''
        Observed_N, All_Set = SetInTime(engine, year, [], ex)
        '''下一个N的观测值'''
        Sp_temp = []
        # 观测总物种生物量占比
        for number in range(3, len(All_Set) + 1):
            Select_Set, Select_ObservedN = SelectInBiomass(engine, year, Observed_N, ex, number)
            Select_Set, Observed_N2 = SelectInBiomass(engine, year + 1, Select_ObservedN, ex, number)
            # 重新选择物种集合；选择物种的观测生物量占比
            D = {}  # 建立矩阵C-P成对字典
            for item in range(100000):  # 生成100000个矩阵
                C_mat = CompetitiveMatrices(Select_ObservedN)
                P_Mat = CproductP(C_mat)
                D[item] = [C_mat, P_Mat]
            Best_Sp, Best_key, P_val = SpearCvsObserved(D, Select_ObservedN, Observed_N2)
            # 最优Spearman值，最优矩阵的位置
            Sp_temp.append(Best_Sp)
        Sp_value.append(Sp_temp)
    file = open("/root/sunqq/txt/Spearman_Variation/2.0.txt", "w")
    np.savetxt(file, Sp_value, delimiter='\n', fmt="%.8f")
    file.close()
    end = time.process_time()
    print('运行时间', end - start)


main()
