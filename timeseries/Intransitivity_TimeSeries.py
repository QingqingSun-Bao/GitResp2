import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from functools import reduce
import time
import warnings
from scipy import stats


warnings.filterwarnings("ignore", category=Warning)
rd = np.random.RandomState(23)



'''各样地观测到的物种'''

def SetInSite(engine,  year, ex):
    Observed_N = []
    sum_D = 0
    for i in year:
        data = pd.read_sql(str(i), con=engine)
        data_site = data.loc[data['样地号'] == str(ex)]
        if len(S_con)==0:
            S_con = set(data_site.loc[:, '物种'])
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

    return Observed_N


'''以生物量筛选物种筛选物种'''


def SelectInBiomass(engine,year, Observed_N, ex):
    D = dict.fromkeys(Observed_N.index.tolist(), 0)
    for key in Observed_N.index.tolist():
        D[key] = Observed_N.loc[key, '2008']
    # 转化成字典
    Select_Set = []
    Select_list = sorted(D.items(), key=lambda x: x[1], reverse=True)  # 按照value值对字典排序(从大到小)
    for i in range(0, len(year) - 1):
        Select_Set.append(Select_list[i][0])
    Select_ObN = SetInTime(engine, year, Select_Set, ex)

    return Select_Set,Select_ObN


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
                pp = reduce(lambda x, y: x * y, np.delete(C_Arr, C_Mat[j, i]))
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
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/nchenjiang?charset=utf8')
    Sp_value = []
    for i in range(104, 105):  # 实验处理
        ex = float(i)
        year = np.linspace(2008, 2009, 1, dtype=int)
        '''物种选择'''

        Select_ObservedN = SetInSite(engine, year, fixed_set, ex)

        # Observed_N = SetInTime(engine, year, [], ex)
        # Select_Set, Select_ObservedN = SelectInBiomass(engine,year,Observed_N, ex)

        '''下一期的N观测值'''
        Observed_N2 = SetInTime(engine, year + 1, fixed_set, ex)

        ''' 生成C-P矩阵'''
        D = {}
        for item in range(100000):  # 生成100000个矩阵
            C_mat = CompetitiveMatrices(Select_ObservedN)
            P_Mat = CproductP(C_mat)
            D[item] = [C_mat, P_Mat]
        ''' 寻找最优C矩阵，计算Spearman值'''
        Best_Sp, Best_key, Best_Pval = SpearCvsObserved(D, Select_ObservedN, Observed_N2)
        Sp_value.append([ex, Best_Sp, Best_Pval])
        '''保存文件'''
        file = open("C:/Users/97899/Desktop/Specise/" + str(ex) + ".txt", "a")
        for i in range(0, 2):
            np.savetxt(file, np.mat(D[Best_key][i]))
        file.close()
    file = open("C:/Users/97899/Desktop/Specise/Spearman.txt", "w")
    np.savetxt(file, Sp_value, delimiter='\n', fmt="%.1f,%.8f,%s,%s")
    file.close()
    end = time.process_time()
    print('运行时间', end - start)

main()
