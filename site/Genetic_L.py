from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
from functools import reduce

'''运算、操作处理相关包'''
import numpy as np
import pandas as pd

rng = np.random.RandomState(19680801)


class GA(object):
    # Mat:含有CP矩阵的字典，数量为一个种群的数量
    # nums初始数据：N*M二维列表；N为进化种群的总数；M为DNA个数(变量的多少)；
    # bound每个变量的边界：M*2的二维列表,[(min,max),(min,max),.....]
    # 运算函数func：为方法对象，可以用def定义，也可以用lambda定义匿名函数
    # DNA的大小：可以指定大小，为None时自动指派，碱基对
    # 染色体交叉的概率，基因变异的概率
    def __init__(self, Mat, bound, Obser_N, func, cross_rate=0.8, mutation=0.1, change=0.05):
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

        # POP_SIZE为进化的种群数
        self.POP_SIZE = len(nums)
        # 一维种群数，二维DNA长度(矩阵元素)
        self.Mat = Mat
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.exchange = change
        self.func = func
        self.Obser_N = Obser_N
        self.global_min = []
        self.fitness = []

    # bound 取值范围
    # POP_SIZE 种群大小
    # copy_POP 复制的种群，用于重置
    # cross_rate 染色体交换概率
    # mutation 基因突变的概率
    # func 适应度函数
    '''将C转化为P矩阵'''

    def CproductP(self, C_Mat):
        n = self.shape[0]
        if type(C_Mat) is not 'numpy.matrix':
            C_Mat = np.mat(C_Mat)
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

    '''获得适应值'''

    def get_fitness(self, C_Mat):
        P_Mat = self.CproductP(C_Mat)
        spman = self.func(P_Mat, self.Obser_N)
        return spman

    '''更新全局最小值'''

    def comparemin(self, C_Mat):
        temp_min = self.get_fitness(C_Mat)
        if temp_min > self.global_min[0]:
            self.global_min[0] = temp_min
            print(self.global_min)
            for i in self.global_min[1]:
                self.fitness[i] = temp_min
                self.Mat[i] = C_Mat

    '''染色体交叉'''

    def crossover(self):
        for C_Mat in self.Mat:
            # 是否进行随机交叉
            if rng.rand() < self.cross_rate:
                # 随机选择交叉的种群
                i_ = rng.randint(0, len(self.Mat), size=1)
                C_Mat = C_Mat + self.Mat[i_[0]]
                # 对应元素修改为0-1
                for i in range(self.shape[0]):
                    for j in range(i, self.shape[0]):
                        if C_Mat[i, j] > 1:
                            C_Mat[i, j] = C_Mat[i, j] - 1
                            C_Mat[j, i] = 1 - C_Mat[i, j]
                        else:
                            C_Mat[j, i] = 1 - C_Mat[i, j]
                self.comparemin(C_Mat)
                # print('交叉调用成功')

    '''基因变异'''

    def mutate(self):
        for C_Mat in self.Mat:
            for i in range(1, C_Mat.shape[0]):
                for j in range(i + 1, C_Mat.shape[0]):
                    if rng.rand() < self.mutation:
                        # 突变
                        C_Mat[i, j] = rng.rand()
                        C_Mat[j, i] = 1 - C_Mat[i, j]
            self.comparemin(C_Mat)
            # print('变异调用成功')

    '''基因互换'''

    def change(self):
        for C_Mat in self.Mat:
            for i in range(1, C_Mat.shape[0]):
                for j in range(i + 1, C_Mat.shape[0]):
                    if rng.rand() < self.exchange:
                        # 变异为对称元素
                        C_Mat[i, j] = 1 - C_Mat[i, j]
                        C_Mat[j, i] = 1 - C_Mat[j, i]
            self.comparemin(C_Mat)

    '''进化'''

    def evolution(self):
        # 整体进化
        self.crossover()
        self.mutate()
        self.change()
        print('完成进化')

    '''一维变量作图'''

    def plot_in_jupyter_1d(self, iter_time=50):
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = 'Simhei'
        fit = []
        Best_CP = []
        max_fitness = 0
        n = 0
        ind_min = []
        for i in range(self.POP_SIZE):
            self.fitness.append(self.get_fitness(self.Mat[i]))
        self.global_min.append(min(self.fitness))
        for index, values in enumerate(self.fitness):
            if values == self.global_min[0]:
                ind_min.append(index)
        self.global_min.append(ind_min)
        while max_fitness <= 0.8:
            max_fitness = max(self.fitness)
            print('最优值',max_fitness)
            index = self.fitness.index(max(self.fitness))
            fit.append(max_fitness)
            Best_CP.append(self.Mat[index])
            self.evolution()
            n += 1
            print(n)
        # for _ in range(iter_time):
        #     print(_)
        #     fitness = self.get_fitness(True)
        #     max_fitness = max(fitness)
        #     index.append(fitness.index(max(fitness)))
        #     fit.append(max_fitness)
        #     print(index[-1])
        #     Best_CP.append(self.Mat[index[-1]])
        #     self.evolution()
        return fit


'''构建函数'''


def func(P_Mat, Observed_N):
    Pre_N = np.dot(P_Mat, Observed_N)  # 做预测值,下一阶段的丰度数据
    df_Observed = pd.DataFrame(Observed_N)
    df_preN = pd.DataFrame(Pre_N)
    spman = []
    for j in range(Observed_N.shape[1]):
        spman_temp = stats.mstats.spearmanr(df_preN.iloc[:, j], df_Observed.iloc[:, j], use_ties=True)
        spman.append(spman_temp[0])

    return np.mean(spman)


if __name__ == "__main__":
    '''产生一个矩阵'''
    C_list = []
    n = 5
    for i in range(100):
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
    ga = GA(C_list, bound, Obser_N, func=func)

    fit = ga.plot_in_jupyter_1d()
    # Spearman
    fig, ax = plt.subplots()
    x1 = np.linspace(1, len(fit), len(fit))
    y1 = np.array(fit)
    ax.plot(x1, y1)
    plt.show()
