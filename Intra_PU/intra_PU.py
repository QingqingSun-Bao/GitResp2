# -*- coding: UTF-8 -*-
import pandas as pd

import numpy as np
from functools import reduce
import time
from scipy import stats
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import datetime
import warnings

warnings.filterwarnings("ignore")

cur = datetime.datetime.now()
rd = np.random.RandomState(23)
pd.set_option("display.max_rows", 1000)
path = "C:/Users/ThinkCentre/Desktop/PU/"
'''筛选物种:各样地组合出现的频繁项集'''


def SelectSpec(path1, year, ex=float(1.0)):
    data = pd.read_excel(path1)
    data_ex = data[data['顺序'] == ex]
    data_site = set(data_ex['样地号'])
    D = {}
    Selsect_set = []
    Site_set = []
    for i in data_site:
        Site_spec = list(set(data_ex[data_ex['样地号'] == i]['物种']))
        Site_set.append(Site_spec)
    te = TransactionEncoder()
    # 进行one-hot编码，0-1
    te_array = te.fit(Site_set).transform(Site_set)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # 用apriori找出频繁项集
    freq = apriori(df, min_support=0.5, use_colnames=True)
    Max_len = 3
    n = 0
    for item in reversed(freq['itemsets']):
        if len(item) >= Max_len:
            # Max_len = len(item)
            Spec_lis = [i for i in item]
            D[n] = []
            Selsect_set.append(Spec_lis)  # 保存所有的物种组合
            D[n].append(Spec_lis)

            n = n + 1
        else:
            break
    # 返回物种集合，以及物种集合所对应的plot
    return Selsect_set,Site_set,data_site


"""根据组合寻找plot"""

def sel_site(key_set,Site_set,data_site):
    site_temp = []
    for i in range(len(Site_set)):
        if (set(Site_set[i]) | set(key_set)) == set(Site_set[i]):
            site_temp.append(list(data_site)[i])
    return site_temp




'''根据频繁项集计算观测到的物种丰度矩阵'''


def ObservedMat(path1, year, Specise, Site):
    dataset = pd.read_excel(path1)
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
        # 求出P的主特征向量U
        eig_value, eig_vector = np.linalg.eig(D[key][1])
        dom_value = eig_value.real.tolist().index(max(eig_value))
        dom_vector = eig_vector[:, dom_value]
        Pre_N = np.dot(D[key][1], dom_vector)  # 做均衡状态下的秩序预测值PU
        arr_preN = np.array(Pre_N.real)
        # df_preN.set_index(Observed_N.index, inplace=True)
        spman = []
        P_val = []
        for j in range(Observed_N.shape[1]):
            spman_temp = stats.mstats.spearmanr(arr_preN, df_Observed.iloc[:, j].values, use_ties=True)
            spman.append(spman_temp[0])
            P_val.append(spman_temp[1])
        average_temp = np.mean(spman)
        if average_temp > Max_Sp:
            Max_Sp = average_temp
            best_key = key

    return Max_Sp, best_key


'''保存文件'''


def savedict(doc, year, datadict, ex):
    file = open(path + "PU_"+str(year)+"/" +str(doc) + "/" + str(year) + '-' + str(ex) + ".txt", "w")
    file.write(str(datadict))
    file.close()


def main():
    start = time.process_time()
    '''参数调整'''
    M = 100000# 循环次数
    # 实验顺序
    for year in range(2009, 2010):  # 年份
        path1 = "C:/Users/ThinkCentre/Desktop/N/" + str(year) + ".xls"
        D_specise = {}
        D_Spvalue = {}
        for ex in range(1, 39):
            ex = float(ex)
            D_Spvalue[ex] = []
            D_specise[ex] = []
            '''频繁项出现的物种组合以及样地号中的物种组合'''
            Select_set, Site_set,data_site= SelectSpec(path1, year, ex)
            # 函数返回所有物种组合，各plot的物种集合，plot的编号集合
            '''拟合各个频繁项集的C-P矩阵,存放在D字典中'''
            D_CP = {}
            for key_Set in Select_set:
                key_Site_set=sel_site(key_Set,Site_set,data_site)
                # 返回组合所在plot的集合
                '''计算观测值，参数：链接，时间，选定物种，物种所在样地'''
                ObservedN = ObservedMat(path1, year, key_Set, key_Site_set)
                # 选出最优的矩阵
                D = {}
                for i in range(M):
                    C_Mat = CompetitiveMat(ObservedN)
                    P_Mat = CproductP(C_Mat)
                    D[i] = [C_Mat, P_Mat]
                Best_Sp, Best_key = SpearCvsObserved(D, ObservedN)
                if Best_Sp >= 0.6:
                    D_CP["0"] = D[Best_key][0], D[Best_key][1]
                    D_specise[ex].append([key_Set, Best_Sp])
                    # print("跳出循环", key_Set,ex)
                    break
            '''保存文件'''
            savedict('CPmat', year, D_CP, ex)
        savedict('Assemb', year, D_specise, 0)
        end = time.process_time()
        print('运行时间', end - start)


main()
