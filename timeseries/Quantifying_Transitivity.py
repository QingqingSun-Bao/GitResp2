import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sympy import *
import math

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

'''导入数据'''


def LoadDataSet(path):
    all_array = np.loadtxt(path)
    C_mat = np.mat(all_array[0:9])

    P_mat = np.mat(all_array[9:18])

    return C_mat, P_mat


'''计算C矩阵的传递性'''


def TransitivityInC(C_mat):
    M = C_mat.shape[0]
    # print(pd.DataFrame(C_mat))
    sum_N = 0
    for i in range(M):
        count = 0
        for j in range(i + 1, M):
            if C_mat[i, j] < C_mat[j, i]:
                count = count + 1
        sum_N = sum_N + count
    Tao_C = (2 * sum_N) / (M * (M - 1))
    if Tao_C > 0.5:
        return Tao_C
    else:
        return 1 - Tao_C


'''计算P矩阵的传递性'''


def TransitivityInP(P_mat):
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
    return Tao_P


'''对物种进行排序'''


def sort_specise(P_mat, specise_set):
    eigenvalue, featurevector = np.linalg.eig(P_mat)
    max_value = 0
    max_index = 0
    eid_value=pd.DataFrame(eigenvalue).astype('float64').values
    for i in range(len(specise_set)):
        if eid_value[i] > max_value:
            max_value = eid_value[i]
            max_index = i
    eig_vect=pd.DataFrame(featurevector).astype('float64')
    # print(np.mat(featurevector))
    vect=eig_vect.iloc[:,max_index].values
    D = {}
    for i in range(len(specise_set)):
        D[specise_set[i]] = vect[i]
    sort_specise=sorted(D.items(), key=lambda x: x[1], reverse=True)
    # print('物种排序',sort_specise)
    return sort_specise


'''实验处理'''


def Exmanage(engine, managelist):
    data = pd.read_sql(str(2008), con=engine)
    exmanage = []
    for i in set(data['顺序']):
        ex_lis = data[data['顺序'] == i].index.tolist()
        exmanage.append(data.iloc[ex_lis[0], managelist])
    exProcess = pd.DataFrame(exmanage)

    return exProcess


'''画氮素和散点图'''


def pltscatter(x, all_process):
    plt.figure()
    group = all_process.groupby('氮素')
    for i in x:
        item = float(i)
        for g in group:
            if g[0] == str(item):
                n = len(g[1].iloc[:, 5])
                x1 = np.repeat(i, n)
                plt.scatter(x1, g[1].iloc[:, 5], c='b', marker='o')
    plt.show()


def Freq_bar(x, all_process):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    group = all_process.groupby('频率')
    g_mean = []
    g_std = []
    for g in group:
        if g[0] != str(0.0):
            g_mean.append(g[1].groupby('氮素').mean())
            g_std.append(g[1].groupby('氮素').std())
    print(g_mean, g_std)
    labels = ['0', '1', '2', '3', '5', '10', '15', '20', '50']
    width = 0.2
    fig, ax = plt.subplots()
    new_ticks = np.linspace(0, 8, 9)
    # 从1-4一共画四个点
    plt.xticks(new_ticks, ['0', '1', '2', '3', '5', '10', '15', '20', '50'])
    ax.bar(labels, g_mean[0].iloc[:, 1], width, yerr=g_std[0].iloc[:, 1], label='2')
    ax.bar(labels, g_mean[1].iloc[:, 1], width, yerr=g_std[1].iloc[:, 1], bottom=g_mean[0].iloc[:, 1], label='12')
    ax.set_ylabel('Tao_P')
    ax.set_xlabel('氮素')
    ax.set_title('频率与氮素共同作用对Tao_P的影响')
    ax.legend()
    plt.show()


def Yige_bar(x, all_process):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    group = all_process.groupby('刈割')
    g_mean = []
    g_std = []
    for g in group:
        g_mean.append(g[1].groupby('氮素').mean())
        g_std.append(g[1].groupby('氮素').std())
    labels = ['0', '1', '2', '3', '5', '10', '15', '20', '50']
    width = 0.2
    fig, ax = plt.subplots()
    new_ticks = np.linspace(0, 8, 9)
    # 从1-4一共画四个点
    plt.xticks(new_ticks, ['0', '1', '2', '3', '5', '10', '15', '20', '50'])
    ax.bar(labels, g_mean[0].iloc[:, 1], width, yerr=g_std[0].iloc[:, 1], label='不刈割')
    ax.bar(labels, g_mean[1].iloc[:, 1], width, yerr=g_std[1].iloc[:, 1], bottom=g_mean[0].iloc[:, 1], label='刈割')
    ax.set_ylabel('Tao_P')
    ax.set_xlabel('氮素')
    ax.legend()
    ax.set_title('刈割与氮素共同作用对Tao_P的影响')
    plt.show()


'''统计重复出现的物种'''


def Re_Specise(path):
    Sp_value = []
    with open(path) as f:
        all_list = f.read().split('\n')
    D = {}
    df = pd.DataFrame()
    ex_specise = {}
    for item in range(len(all_list) - 1):
        item_list = all_list[item].split(',')
        Sp_value.append(item_list[1])
        ex_temp = []
        for i in range(2, 11):
            item_list[i] = item_list[i].replace("'", '')
            item_list[i] = item_list[i].replace('[', '')
            item_list[i] = item_list[i].replace(']', '')
            ex_temp.append(item_list[i])
            if item_list[i] not in D.keys():
                D[item_list[i]] = 1
            else:
                D[item_list[i]] = D[item_list[i]] + 1
        ex_specise[item + 1] = ex_temp
        df = pd.DataFrame(ex_specise).T
        print(df)

    return Sp_value, df


def main():
    path = 'C:/Users/97899/Desktop/Biomass/Spearman.txt'
    Sp, ex_specise = Re_Specise(path)
    Tao_C = []
    Tao_PP = []
    # 各实验处理
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    ex_manage = [5, 6, 7, 8]
    # 顺序，氮素，频率，
    all_process = Exmanage(engine, ex_manage)
    for i in range(1, 39):
        ex = float(i)
        path = 'C:/Users/97899/Desktop/Biomass/' + str(ex) + '.txt'
        C_mat, P_mat = LoadDataSet(path)
        sort=sort_specise(P_mat, ex_specise.iloc[i - 1, :])
        # print(ex,sort)
        Tao_C.append(TransitivityInC(C_mat))
        Tao_PP.append(TransitivityInP(P_mat))
    all_process['Tao_C'] = Tao_C
    all_process['Tao_PP'] = Tao_PP
    print(all_process)
    # file = open("C:/Users/97899/Desktop/Biomass/all_process.txt", "w")
    # np.savetxt(file, all_process,fmt=('%.8f, %.8f, %.8f, %.8f, %.8f, %.8,f'))
    # file.close()
    x = np.array([0, 1, 2, 3, 5, 10, 15, 20, 50])
    # pltscatter(x,all_process)
    # Freq_bar(x, all_process)
    # Yige_bar(x, all_process)


main()
