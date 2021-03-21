# -*- coding: UTF-8 -*-
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from functools import reduce
import time
from scipy import stats
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from GeneticJYZM import GA

rd = np.random.RandomState(23)
pd.set_option("display.max_rows", 1000)
path = "C:/Users/97899/Desktop/12.2/"
'''筛选物种:各样地组合出现的频繁项集'''


def SelectSpec(engine, year, ex=str(1.0)):
    data = pd.read_sql(str(year), con=engine)
    data_ex = data[data['顺序'] == ex]
    data_site = set(data_ex['样地号'])
    D = {}
    Selsect_set = []
    Spec_set = []
    for i in data_site:
        Site_spec = list(set(data_ex[data_ex['样地号'] == i]['物种']))
        Spec_set.append(Site_spec)
    te = TransactionEncoder()
    # 进行one-hot编码，0-1
    te_array = te.fit(Spec_set).transform(Spec_set)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # 用apriori找出频繁项集
    freq = apriori(df, min_support=0.5, use_colnames=True)
    Max_len = 0
    n = 0
    for item in reversed(freq['itemsets']):
        if len(item) >= Max_len:
            Max_len = len(item)
            Spec_lis = [i for i in item]
            D[n] = []
            Selsect_set.append(Spec_lis)
            D[n].append(Spec_lis)
            site_temp = []
            for i in range(len(Spec_set)):
                if (set(Spec_set[i]) | item) == set(Spec_set[i]):
                    site_temp.append(list(data_site)[i])
            D[n].append(site_temp)
            n = n + 1
        else:
            break
    # print(D)
    return Selsect_set, D


'''根据频繁项集计算观测到的物种丰度矩阵'''


def ObservedMat(engine, year, Specise, Site):
    dataset = pd.read_sql(str(year), con=engine)
    ObservedN = []
    for item in Site:
        data = dataset[dataset['样地号'] == item]
        data.set_index(['物种'], inplace=True)
        observed_temp = []
        for j in Specise:
            observed_temp.append(float(data.loc[j, '干重g']))
        sum_0 = sum(np.array(observed_temp))
        observed_temp = [i / sum_0 for i in observed_temp]
        ObservedN.append(observed_temp)
    df = pd.DataFrame(ObservedN, columns=Specise).T

    return df


'''产生大量的竞争矩阵C'''


# 全局变量
def CompetitiveMat(Observed_N):
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
                C_Arr = np.array(C_Mat[j,])[0]
                pp = reduce(lambda x, y: x * y, np.delete(C_Arr, i))
                pp = pow(pp, 1 / (n - 2))
                # 几何平均值
                Geome_series = [pow(pp, i) for i in range(n - 1)]
                P[i, j] = 1 / (n - 1) * C_Mat[i, j] * reduce(lambda l, k: l + k, Geome_series)
                # 近似式，累加和
    return P


'''利用Spearman寻找最优的矩阵C'''


def SpearCvsObserved(D, Observed_N):
    Max_Sp = 0
    best_key = 0
    best_P = []
    df_Observed = pd.DataFrame(Observed_N)
    for key in D.keys():
        Pre_N = np.dot(D[key][1], Observed_N)  # 做预测值,下一阶段的丰度数据
        df_preN = pd.DataFrame(Pre_N)
        df_preN.set_index(Observed_N.index, inplace=True)
        spman = []
        P_val = []
        for j in range(Observed_N.shape[1]):
            spman_temp = stats.mstats.spearmanr(df_preN.iloc[:, j], df_Observed.iloc[:, j], use_ties=True)
            spman.append(spman_temp[0])
            P_val.append(spman_temp[1])
        average_temp = np.mean(spman)
        if average_temp > Max_Sp:
            Max_Sp = average_temp
            best_key = key
            best_P = P_val

    return Max_Sp, best_key, best_P


'''产生基因池'''


def genepool(Observed_N):
    C_list = []
    for i in range(100):
        C_Mat = CompetitiveMat(Observed_N)
        C_list.append(C_Mat)

    return C_list


'''利用遗传算法进行优化'''


def gene(C_list, Observed_N, func):
    n = Observed_N.shape[0]
    bound = [(0, 1)] * n * n
    ga = GA(C_list, bound, Observed_N, func=func, rng=rd, DNA_SIZE=len(bound))
    fit, fit_c, fit_pv = ga.plot_in_jupyter_1d()
    return fit[-1], fit_c[-1], fit_pv[-1]


'''构建函数'''


def func(P_Mat, Observed_N):
    Pre_N = np.dot(P_Mat, Observed_N)  # 做预测值,下一阶段的丰度数据
    df_Observed = pd.DataFrame(Observed_N)
    df_preN = pd.DataFrame(Pre_N)
    sp = []
    P_val = []
    for j in range(Observed_N.shape[1]):
        sp_temp = stats.mstats.spearmanr(df_preN.iloc[:, j], df_Observed.iloc[:, j], use_ties=True)
        sp.append(sp_temp[0])
        P_val.append(sp_temp[1])
    return np.mean(sp), P_val


'''保存文件'''


def savedict(doc, year, datadict, ex):
    file = open(path + str(doc) + "/" + str(year) + '-' + str(ex) + ".txt", "w")
    file.write(str(datadict))
    file.close()


def main():
    start = time.process_time()
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')

    '''参数调整'''
    M = 100000  # 循环次数

    # 实验顺序
    for year in range(2008, 2009):  # 年份
        D_specise = {}
        D_Spvalue = {}
        for ex in range(1, 2):
            ex = str(float(ex))
            D_Spvalue[ex] = []
            D_specise[ex] = []
            '''频繁项出现的物种组合以及样地号'''
            Select_set, Site_set = SelectSpec(engine, year, ex)

            '''拟合各个频繁项集的C-P矩阵,存放在D字典中'''
            D_CP = {}
            D = {}
            for key_Set in Site_set:
                '''计算观测值，参数：链接，时间，选定物种，物种所在样地'''
                ObservedN = ObservedMat(engine, year, Site_set[key_Set][0], Site_set[key_Set][1])
                for i in range(M):
                    C_Mat = CompetitiveMat(ObservedN)
                    P_Mat = CproductP(C_Mat)
                    D[i] = [C_Mat, P_Mat]
                Best_Sp, Best_key, Best_P = SpearCvsObserved(D, ObservedN)
                # 最优的Spearman值，字典中的位置，最优P值
                if Best_Sp < 0.7:
                    Clist = genepool(ObservedN)
                    Best_Sp, Best_C, Best_Pv = gene(Clist, ObservedN, func)
                    print('应用遗传算法')
                    Best_P = CproductP(np.mat(Best_C))
                    D_CP[key_Set] = [Best_C, Best_P]
                    D_Spvalue[ex].append([Best_Sp, Best_Pv])
                else:
                    D_CP[key_Set] = [D[Best_key][0], D[Best_key][1]]
                    D_Spvalue[ex].append([Best_Sp, Best_P])
                D_specise[ex].append(Site_set[key_Set][0])
            '''保存文件'''
            savedict('CPmat', year, D_CP, ex)
        savedict("Spearman", year, D_Spvalue, 0)
        savedict('Assemb', year, D_specise, 0)
        end = time.process_time()
        print('运行时间', end - start)


main()
