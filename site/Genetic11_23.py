from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
from functools import reduce

'''运算、操作处理相关包'''
import numpy as np
import pandas as pd

rng = np.random.RandomState(19680801)


class GA(object):
    def __init__(self, Mat, bound, Obser_N, func, DNA_SIZE=None, cross_rate=0.9, mutation=0.1, change=0.7):
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

        self.POP_SIZE = len(nums)
        # 一维种群数，二维DNA长度(矩阵元素)
        self.Mat = Mat
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.exchange = change
        self.func = func
        self.Obser_N = Obser_N
        self.trans = []
        self.global_best = [Mat[0], 0]
        self.fitness = []

    '''将C转化为P矩阵'''

    def CproductP(self, C_list):
        n = self.shape[0]
        P_list = []
        for C_Mat in C_list:
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
            P_list.append(P)
        return P_list

    '''获得适应值(目标函数最大值)'''

    def get_fitness(self, non_single=False):
        average_SP = []
        P_list = self.CproductP(self.Mat)
        for P_Mat in P_list:
            spman = self.func(P_Mat, self.Obser_N)
            average_SP.append(spman)
        fitness = list(average_SP)
        self.fitness = fitness
        # 精英策略
        max_index = self.fitness.index(max(self.fitness))
        if max(fitness) > self.global_best[1]:
            self.global_best = [self.Mat[max_index], self.fitness[max_index]]
        # print(average_SP)
        return fitness

    '''自然选择'''

    def select(self):
        fitness = self.fitness
        min_fit = min(fitness)
        fit_var = list(np.array(fitness) - min_fit+0.001)
        fitness.sort(reverse=True)
        self.Mat = np.array(self.Mat)[rng.choice(np.arange(len(self.Mat)), size=len(self.Mat), replace=True,
                                                 p=fit_var / np.sum(fit_var))]
        '''更换选择方法:后25%翻倍，前50%直接保留'''
        # fit_sort = sorted(enumerate(self.fitness), key=lambda x: x[1], reverse=True)
        # fit_sortIndex = [x[0] for x in fit_sort]
        # index = fit_sortIndex[:int(self.POP_SIZE * 0.25)]
        # # 前25%
        # index2 = fit_sortIndex[-int(self.POP_SIZE * 0.25):]
        # # 后25%
        # index3=fit_sortIndex[-int(self.POP_SIZE * 0.5):-int(self.POP_SIZE * 0.25)]
        # # 后25%-50%
        # index.extend(index2)
        # index.extend(index2)
        # index.extend(index3)
        # # 完全复制的个体
        # Mat_list = [self.Mat[i] for i in index]
        # self.Mat = Mat_list

    '''染色体交叉'''

    def crossover(self):
        for C_Mat in self.Mat:
            if rng.rand() < self.cross_rate:
                # 随机选择交叉的种群
                i_ = rng.randint(0, len(self.Mat), size=1)
                C_Mat = C_Mat + self.Mat[i_[0]][0]
                # 对应元素修改为0-1
                for i in range(self.shape[0]):
                    for j in range(i, self.shape[0]):
                        if C_Mat[i, j] > 1:
                            C_Mat[i, j] = C_Mat[i, j] - 1
                            C_Mat[j, i] = 1 - C_Mat[i, j]
                        else:
                            C_Mat[j, i] = 1 - C_Mat[i, j]

    '''基因变异'''

    def mutate(self):
        for C_Mat in self.Mat:
            for i in range(1, C_Mat.shape[0]):
                for j in range(i + 1, C_Mat.shape[0]):
                    if rng.rand() < self.mutation:
                        # 突变
                        C_Mat[i, j] = rng.rand()
                        C_Mat[j, i] = 1 - C_Mat[i, j]

    '''基因互换'''

    def change(self):
        for C_Mat in self.Mat:
            for i in range(1, C_Mat.shape[0]):
                for j in range(i + 1, C_Mat.shape[0]):
                    if rng.rand() < self.exchange:
                        # 变异为对称元素
                        C_Mat[i, j] = 1 - C_Mat[i, j]
                        C_Mat[j, i] = 1 - C_Mat[j, i]

    '''进化'''

    def evolution(self):
        # 整体进化
        # 依据全局的fitness进行基因的筛选
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
        index = fit_sortIndex[:int(self.POP_SIZE * 0.25)]
        index2 = fit_sortIndex[-int(self.POP_SIZE * 0.25):]
        index.extend(index2)
        # 完全复制的个体
        Mat_list = [self.Mat[i] for i in index]
        C_lst = self.ProC(self.POP_SIZE - len(index))
        # print('生成的C矩阵长度', len(C_lst))
        Mat_list.extend(C_lst)
        self.Mat = Mat_list
        print('完成灾变', len(Mat_list))

    '''随机产生矩阵'''

    def ProC(self, m):
        n = self.shape[0]
        C_list = []
        for _ in range(m):
            mat_uniform = rng.uniform(0, 1, n)
            Mat_tu = np.mat(np.triu(mat_uniform, 1))
            mat_tl = np.mat(np.mat(np.ones(n)) - np.mat(Mat_tu.T))
            Mat_Tl = np.tril(mat_tl, 0)
            C_mat = Mat_Tl + Mat_tu
            C_list.append(C_mat)
        return C_list

    '''统计最大值出现的次数'''

    def count_fit(self, fitness):
        D = {}
        for fit in fitness:
            if fit in D.keys():
                D[fit] += 1
            else:
                D[fit] = 1
        fit_fre = sorted(D.items(), key=lambda x: x[1], reverse=True)[0][1]
        return fit_fre

    '''一维变量作图'''

    def plot_in_jupyter_1d(self, iter_time=250):
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = 'Simhei'
        fit = []; index = []; Best_CP = []
        max_fitness = 0
        number = 1
        D = {}
        N = round(iter_time/15)
        for number in range(1,iter_time):
            print(number)
            fitness = self.get_fitness(True)  # 随即确定全局fitness
            # 适应度最高的个体代替适应度最低的个体
            min_index = fitness.index(min(fitness))
            self.Mat[min_index] = self.global_best[0]
            self.fitness[min_index] = self.global_best[1]
            max_fit_ind = fitness.index(max(fitness))
            index.append(max_fit_ind)
            fit.append(max(fitness))
            max_fitness=max(fitness)
            # print(fit)
            if number % 15 == 0:
                # print('是15的整数倍')
                disaster_number=self.count_fit(fit)
                if disaster_number > 6:
                    self.disaster()
                # if disaster_number==15:
                #     self.mutation = self.mutation + 1 / 10 * 1 / (1 + np.exp((iter_time - number) / (iter_time - N)))
                #     print('基因突变概率变换',self.mutation)
            Best_CP.append(self.Mat[max_fit_ind])
            self.evolution()
            number += 1
            if max_fitness>0.9:
               break
        return fit[-1]


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
    for item in range(100):
        matrix_uniform = rng.uniform(0, 1, (n, n))
        Mat_Triu = np.mat(np.triu(matrix_uniform, 1))
        mat_tril = np.mat(np.mat(np.ones((n, n))) - np.mat(Mat_Triu.T))
        Mat_Tril = np.tril(mat_tril, 0)
        C_mat = Mat_Tril + Mat_Triu
        C_list.append(C_mat)
    # 将矩阵拉伸成向量
    Obser_N = np.mat(np.array([0.350255, 0.321227, 0.483453, 0.231161, 0.138150,
                               0.460624, 0.263205, 0.079081, 0.018303, 0.060069,
                               0.036536, 0.358654, 0.161724, 0.622763, 0.581023,
                               0.000644, 0.018935, 0.061092, 0.014886, 0.008251,
                               0.122021, 0.017938, 0.070977, 0.029829, 0.1810645]).reshape((5, 5)))
    m = Obser_N.shape[0]
    bound = [(0, 1)] * m * m
    l=np.arange(0,1,0.1)
    zuhe=[]
    for item_cro in l:
        for item_mu in l:
            for item_ch in l:
                ga = GA(C_list, bound, Obser_N, func=func, DNA_SIZE=len(bound),cross_rate=item_cro,
                        mutation=item_mu,change=item_ch)
                fit = ga.plot_in_jupyter_1d()
                zuhe.append([item_cro,item_mu,item_ch,fit])
    print(zuhe)





    # # Spearman
    # fig, ax = plt.subplots()
    # x1 = np.linspace(1, len(fit), len(fit))
    # y1 = np.array(fit)
    # ax.plot(x1, y1)
    # plt.show()
