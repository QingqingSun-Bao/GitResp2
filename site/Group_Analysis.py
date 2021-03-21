import pandas as pd
import numpy as np
from GeneticJYZM import GA
import matplotlib.pyplot as plt
from scipy import stats

# path='C:/Users/97899/Desktop/N_2009/环数_2009.xls'
# pd.set_option('display.max_rows',1000)
# pd.set_option('display.max_columns',1000)
# df=pd.read_excel(path,sheet_name='2009')
# print(df.iloc[:,1:6])
# df_de=df.iloc[:,1:7].groupby('氮素').describe()
# print(df_de)
# fr = open('C:/Users/97899/Desktop/N_2009/Spearman/2009-0.txt',encoding='utf-8')
# dic = eval(fr.read())  # 将str转化成dict
# fr.close()
# # for key in dic:
# #     print(key,dic[key][0])

rng = np.random.RandomState(19680801)
def func(P_Mat, Observed_N):
    Pre_N = np.dot(P_Mat, Observed_N)  # 做预测值,下一阶段的丰度数据
    df_Observed = pd.DataFrame(Observed_N)
    df_preN = pd.DataFrame(Pre_N)
    spman = []
    for j in range(Observed_N.shape[1]):
        spman_temp = stats.mstats.spearmanr(df_preN.iloc[:, j], df_Observed.iloc[:, j], use_ties=True)
        spman.append(spman_temp[0])
    return np.mean(spman)

C_list = []
n = 5
for i in range(100):
    print(rng)
    matrix_uniform = rng.uniform(0, 1, (n, n))
    Mat_Triu = np.mat(np.triu(matrix_uniform, 1))
    mat_tril = np.mat(np.mat(np.ones((n, n))) - np.mat(Mat_Triu.T))
    Mat_Tril = np.tril(mat_tril, 0)
    C_Mat = Mat_Tril + Mat_Triu
    C_list.append(C_Mat)
# 将矩阵拉伸成向量
Obser_N = np.mat(np.array([0.350255, 0.321227, 0.483453, 0.231161, 0.138150,
                           0.460624, 0.263205, 0.079081, 0.018303, 0.060069,
                           0.036536, 0.358654, 0.161724, 0.622763, 0.581023,
                           0.000644, 0.018935, 0.061092, 0.014886, 0.008251,
                           0.122021, 0.017938, 0.070977, 0.029829, 0.1810645]).reshape((5, 5)))
m = Obser_N.shape[0]
bound = [(0, 1)] * m * m
ga= GA(C_list, bound, Obser_N, func=func, rng=rng,DNA_SIZE=len(bound))

fit,fit_mat= ga.plot_in_jupyter_1d()
# Spearman
print(fit[-1],fit_mat[-1])
fig, ax = plt.subplots()
x1 = np.linspace(1, len(fit), len(fit))
y1 = np.array(fit)
ax.plot(x1, y1)
plt.show()