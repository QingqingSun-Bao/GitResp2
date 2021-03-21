from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
from functools import reduce

'''运算、操作处理相关包'''
import numpy as np
import pandas as pd


class GA(object):
    def __init__(self, Mat, bound, Obser_N, func, rng, DNA_SIZE=None, cross_rate=0.8, mutation=0.03, change=0.05):
        nums = Mat[0].ravel()
        for i in range(1, len(Mat)):
            nums_1 = Mat[i].ravel()
            nums = np.vstack((nums, nums_1))
        nums = np.array(nums)
        bound = np.array(bound)
        self.bound = bound
        self.shape = Mat[1].shape
        if nums.shape[1] != bound.shape[0]:
            # 变量个数与变量范围个数不一致
            raise Exception(f'范围的数量与变量的数量不一致，您有{nums.shape[1]}个变量，却有{bound.shape[0]}个范围')
        for var in nums:
            # 取其中一条数据
            for index, var_curr in enumerate(var):
                # enumerate(),将数据与数据下标组合在一起
                if var_curr < bound[index][0] or var_curr > bound[index][1]:
                    raise Exception(f'{var_curr}不在取值范围内')

        self.DNA_SIZE = DNA_SIZE
        # POP_SIZE为进化的种群数
        self.POP_SIZE = len(nums)
        # 一维种群数，二维DNA长度(矩阵元素)
        self.rng = rng
        self.Mat = Mat
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.exchange = change
        self.func = func
        self.Obser_N = Obser_N
        self.trans = []
        self.global_best = [Mat[0], 0, 0]
        self.fitness = []
        self.pv = []

    '''将C转化为P矩阵'''

    def CproductP(self, C_List):
        P_list = []
        for C_mat in C_List:
            if type(C_mat) is not 'numpy.matrix':
                C_mat = np.mat(C_mat)
            P = np.mat(np.zeros(shape=self.shape))
            for i_ in range(self.shape[0]):
                for j in range(self.shape[0]):
                    if i_ == j:
                        C_Arr = []
                        for item in range(self.shape[0]):
                            C_Arr.append(C_mat[i_, item])
                        P[i_, i_] = reduce(lambda x, y: x * y, C_Arr)
                        # 矩阵某行的连乘积reduce()
                    else:
                        C_Arr = np.array(C_mat[j,])[0]
                        pp = reduce(lambda x, y: x * y, np.delete(C_Arr, i_))
                        pp = pow(pp, 1 / (self.shape[0] - 2))
                        # 几何平均值
                        Geo_series = [pow(pp, x) for x in range(self.shape[0] - 1)]
                        P[i_, j] = 1 / (self.shape[0] - 1) * C_mat[i_, j] * reduce(lambda l, k: l + k, Geo_series)
                        # 近似式，累加和
            P_list.append(P)
        return P_list

    '''获得适应值(目标函数最大值)'''

    def get_fitness(self, non_single=False):
        average_SP = []
        average_PV = []
        P_list = self.CproductP(self.Mat)
        for P_Mat in P_list:
            sp, pv = self.func(P_Mat, self.Obser_N)
            average_SP.append(sp)
            average_PV.append(pv)
        fitness = average_SP
        self.fitness = fitness
        self.pv = average_PV
        # 精英策略
        max_index = self.fitness.index(max(self.fitness))
        if max(fitness) > self.global_best[1]:
            self.global_best = [self.Mat[max_index], self.fitness[max_index], self.pv[max_index]]
        # 最优值相同的情况下，换取P值最显著的
        if max(fitness) == self.global_best[1]:
            m1 = len([x for x in self.global_best[2] if x < 0.05])
            m2 = len([y for y in average_PV[max_index] if y < 0.05])
            if m1 < m2:
                self.global_best = [self.Mat[max_index], self.fitness[max_index], self.pv[max_index]]

        return fitness

    '''自然选择'''

    def select(self):
        fitness = self.fitness
        min_fit = min(fitness)
        fit_var = list(np.array(fitness) - min_fit)
        fitness.sort(reverse=True)
        self.Mat = np.array(self.Mat)[self.rng.choice(np.arange(len(self.Mat)), size=len(self.Mat), replace=True,
                                                      p=fit_var / np.sum(fit_var))]

    '''染色体交叉'''

    def crossover(self):
        for C_mat in self.Mat:
            if self.rng.rand() < self.cross_rate:
                # 随机选择交叉的种群
                i_ = self.rng.randint(0, len(self.Mat), size=1)
                C_mat = C_mat + self.Mat[i_[0]][0]
                # 对应元素修改为0-1
                for i in range(self.shape[0]):
                    for j in range(i, self.shape[0]):
                        if C_mat[i, j] > 1:
                            C_mat[i, j] = C_mat[i, j] - 1
                            C_mat[j, i] = 1 - C_mat[i, j]
                        else:
                            C_mat[j, i] = 1 - C_mat[i, j]

    '''基因变异'''

    def mutate(self):
        for C_mat in self.Mat:
            for i in range(1, C_mat.shape[0]):
                for j in range(i + 1, C_mat.shape[0]):
                    if self.rng.rand() < self.mutation:
                        # 突变
                        C_mat[i, j] = self.rng.rand()
                        C_mat[j, i] = 1 - C_mat[i, j]

    '''基因互换'''

    def change(self):
        for C_Mat in self.Mat:
            for i in range(1, C_Mat.shape[0]):
                for j in range(i + 1, C_Mat.shape[0]):
                    if self.rng.rand() < self.exchange:
                        # 变异为对称元素
                        C_Mat[i, j] = 1 - C_Mat[i, j]
                        C_Mat[j, i] = 1 - C_Mat[j, i]

    '''进化'''

    def evolution(self):
        # 整体进化
        self.select()
        self.crossover()
        self.mutate()
        self.change()
        print('完成进化')

    '''灾变'''

    def disaster(self):
        # 前25%直接复制，后25%直接复制,删除中间50%的优秀个体，
        # 只产生50%的新个体进行补充
        fit_sort = sorted(enumerate(self.fitness), key=lambda x: x[1], reverse=True)
        fit_sortIndex = [x[0] for x in fit_sort]
        index = fit_sortIndex[:int(self.POP_SIZE * 0.35)]
        index2 = fit_sortIndex[-int(self.POP_SIZE * 0.15):]
        index.extend(index2)
        # 完全复制的个体
        Mat_list = [self.Mat[i] for i in index]
        C_lst = self.ProC(self.POP_SIZE - len(index))
        # print('生成的C矩阵长度', len(C_lst))
        Mat_list.extend(C_lst)
        self.Mat = Mat_list
        # print('完成灾变', len(Mat_list))

    '''随机产生矩阵'''

    def ProC(self, number):
        shape = self.shape[0]
        C_List = []
        for _ in range(number):
            mat_uniform = self.rng.uniform(0, 1, shape)
            Mat_tu = np.mat(np.triu(mat_uniform, 1))
            mat_tl = np.mat(np.mat(np.ones(shape)) - np.mat(Mat_tu.T))
            Mat_Tl = np.tril(mat_tl, 0)
            C_mat = Mat_Tl + Mat_tu
            C_List.append(C_mat)
        return C_List

    '''统计最大值出现的次数'''

    def count_fit(self, fitness):
        D = {}
        for fit_ in fitness:
            if fit_ in D.keys():
                D[fit_] += 1
            else:
                D[fit_] = 1
        fit_fre = sorted(D.items(), key=lambda x: x[1], reverse=True)[0][1]
        return fit_fre

    '''一维变量作图'''

    def plot_in_jupyter_1d(self):
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = 'Simhei'
        fit1 = []
        Best_CP = []
        Best_pv = []
        max_fitness = 0
        number = 1
        while max_fitness <= 0.79:
            # print(number)
            fitness = self.get_fitness(True)  # 随即确定全局fitness,global_best
            # 适应度最高的个体代替适应度最低的个体
            min_index = fitness.index(min(fitness))
            self.Mat[min_index] = self.global_best[0]
            self.fitness[min_index] = self.global_best[1]
            self.pv[min_index] = self.global_best[2]
            max_fitness = max(fitness)
            max_fit_ind = fitness.index(max_fitness)
            fit1.append(max(fitness))
            # 取最优值对应的矩阵以及P值
            Best_pv.append(self.pv[max_fit_ind])
            Best_CP.append(self.Mat[max_fit_ind])
            # print(fit1)
            # print(self.pv[max_fit_ind])
            if number % 15 == 0:
                # print('是15的整数倍')
                if self.count_fit(fit1) > 6:
                    self.disaster()
            self.evolution()
            number += 1
            if number == 10:
                break
        return fit1, Best_CP, Best_pv


'''构建函数'''


def con_func(P_Mat, Observed_N):
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


if __name__ == "__main__":
    '''产生一个矩阵'''
    rd = np.random.RandomState(19680801)
    C_list = []
    n = 5
    for item in range(100):
        print(rd)
        matrix_uniform = rd.uniform(0, 1, (n, n))
        Mat_Triu = np.mat(np.triu(matrix_uniform, 1))
        mat_tril = np.mat(np.mat(np.ones((n, n))) - np.mat(Mat_Triu.T))
        Mat_Tril = np.tril(mat_tril, 0)
        C_Mat = Mat_Tril + Mat_Triu
        C_list.append(C_Mat)
    # 将矩阵拉伸成向量
    Observed_N = np.mat(np.array([0.350255, 0.321227, 0.483453, 0.231161, 0.138150,
                                  0.460624, 0.263205, 0.079081, 0.018303, 0.060069,
                                  0.036536, 0.358654, 0.161724, 0.622763, 0.581023,
                                  0.000644, 0.018935, 0.061092, 0.014886, 0.008251,
                                  0.122021, 0.017938, 0.070977, 0.029829, 0.1810645]).reshape((5, 5)))
    m = Observed_N.shape[0]
    bound = [(0, 1)] * m * m
    ga = GA(C_list, bound, Observed_N, func=con_func, rng=rd, DNA_SIZE=len(bound))

    fit, fit_c, fit_p = ga.plot_in_jupyter_1d()
    print(fit[-1])
    print(fit_c[-1])
    # Spearman
    fig, ax = plt.subplots()
    x1 = np.linspace(1, len(fit), len(fit))
    y1 = np.array(fit)
    ax.plot(x1, y1)
    plt.show()
