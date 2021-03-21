from matplotlib import pyplot as plt
import matplotlib
'''运算、操作处理相关包'''
import numpy as np
import pandas as pd


class GA(object):
    # nums初始数据：N*M二维列表；N为进化种群的总数；M为DNA个数(变量的多少)；
    # bound每个变量的边界：M*2的二维列表,[(min,max),(min,max),.....]
    # 运算函数func：为方法对象，可以用def定义，也可以用lambda定义匿名函数
    # DNA的大小：可以指定大小，为None时自动指派，碱基对
    # 染色体交叉的概率，基因变异的概率
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        nums = np.array(nums)
        print(np.shape(nums))
        bound = np.array(bound)
        self.bound = bound
        if nums.shape[1] != bound.shape[0]:
            # 变量个数与变量范围个数不一致
            raise Exception(f'范围的数量与变量的数量不一致，您有{nums.shape[1]}个变量，却有{bound.shape[0]}个范围')
        for var in nums:
            # 取其中一条数据
            for index, var_curr in enumerate(var):
                # enumerate(),将数据与数据下标组合在一起
                if var_curr < bound[index][0] or var_curr > bound[index][1]:
                    raise Exception(f'{var_curr}不在取值范围内')
        for min_bound, max_bound in bound:
            if max_bound < min_bound:
                raise Exception(f'抱歉，({min_bound},{max_bound})不是合格的取值范围')
        # var_len 为所有变量的取值范围大小
        # bit为每个变量按整数编码最小的二进制位数
        min_nums, max_nums = np.array(list(zip(*bound)))
        # 将元组数据解压成二维数据列表
        self.var_len = var_len = max_nums - min_nums
        bits = np.ceil(np.log2(var_len + 1))
        # 向上取整，做成二进制数

        if DNA_SIZE == None:
            DNA_SIZE = int(np.max(bits))
            # 自动指派DNA大小，取最长区间作为DNA的大小，碱基对
        self.DNA_SIZE = DNA_SIZE

        # POP_SIZE为进化的种群数
        self.POP_SIZE = len(nums)
        POP = np.zeros((*nums.shape, DNA_SIZE))
        # 初始化
        # POP 编码后的种群，一维是种群数量，二维是各个DNA，三维是碱基对
        for i in range(nums.shape[0]):
            for j in range(nums.shape[1]):
                # 编码方式
                num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
                # 用format格式化,以0补充二进制字符串，然后拆分成列表
                POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        self.POP = POP
        # 用于后面重置(reset)
        self.copy_POP = POP.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func

    # save args 对象保留参数:
    # bound 取值范围
    # var_len 取值范围大小
    # POP_SIZE 种群大小
    # copy_POP 复制的种群，用于重置
    # cross_rate 染色体交换概率
    # mutation 基因突变的概率
    # func 适应度函数

    '''将编码后的DNA翻译回来(解码)'''

    def translateDNA(self):
        # 先将二进制列表转化为10进制数
        W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE))[::-1]
        binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
        # 并将三维数组转化为两维向量
        # 再通过解码公式解码
        for i in range(binary_vector.shape[0]):
            for j in range(binary_vector.shape[1]):
                binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
                binary_vector[i, j] += self.bound[j][0]
        return binary_vector

    '''获得适应值(目标函数最大值)'''

    def get_fitness(self, non_negative=False):
        # 调用DNA解码返回二维向量(种群，变量值(DNA))
        result = self.func(*np.array(list(zip(*self.translateDNA()))))
        # 若需要适应值非负值的限制
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        # 返回一维向量
        return result

    '''自然选择'''

    def select(self):
        fitness = self.get_fitness(non_negative=True)
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True, p= fitness / np.sum(fitness))]
        # np.random.choice，对序列编号随机抽取，P为元素列表被选中的概率列表

    '''染色体交叉'''

    def crossover(self):
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                # 随机确定交叉种群
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                # 随机确定要交叉的碱基对
                cross_points = np.random.randint(0, 2, size=(len(self.var_len), self.DNA_SIZE)).astype(np.bool)
                # 将碱基对交给被交叉种群
                people[cross_points] = self.POP[i_, cross_points]

    '''基因变异'''

    def mutate(self):
        for people in self.POP:
            for var in people:
                for point in range(self.DNA_SIZE):
                    if np.random.rand() < self.mutation:
                        if var[point] == 0 :
                            var[point] = 1
                        else:
                            var[point] = 0

    '''进化'''

    def evolution(self):
        # 整体进化
        self.select()
        self.crossover()
        self.mutate()

    '''重置'''

    def reset(self):
        self.POP = self.copy_POP.copy()
        # 将内存备份的copy_POP复制回来即可

    '''打印当前状态日志'''

    def log(self):
        return pd.DataFrame(np.hstack((self.translateDNA(), self.get_fitness().reshape((len(self.POP), 1)))),
                            columns=[f'x{i}' for i in range(len(self.var_len))] + ['F'])
        # np.hstack 在水平方向上堆叠

    '''一维变量作图'''

    def plot_in_jupyter_1d(self, iter_time=200):
        is_ipython = 'inline' in matplotlib.get_backend()
        # 利用'inline'激活动态显示功能
        if is_ipython:
            from IPython import display
        plt.ion()
        # 启动交互式
        plt.rcParams['axes.unicode_minus']=False
        plt.rcParams['font.sans-serif']='Simhei'
        fit=[]
        for _ in range(iter_time):
            plt.cla()
            x = np.linspace(*self.bound[0], self.var_len[0] * 50)
            plt.title('第' + str(_) + '次迭代')
            plt.plot(x, self.func(x))
            x = self.translateDNA().reshape(self.POP_SIZE)
            plt.scatter(x, self.func(x), s=200, lw=0, c='red', alpha=0.5)
            plt.pause(0.01)
            # if is_ipython:
            #  # 清除输出并显示最新的图片
            #     display.clear_output(wait=True)
            #     display.display(plt.gcf())
            self.evolution()
            fit.append(self.log().max().tolist()[1])
        plt.ioff()
        return fit



if __name__ == "__main__":
    '''单变量'''
    func = lambda x: np.sin(10 * x) * x + np.cos(2 * x) * x
    nums=[[np.random.rand()*5] for _ in range(100)]
    ga = GA(nums,[(0,5)],DNA_SIZE=11,func=func)
    fit=ga.plot_in_jupyter_1d()
    fig,ax=plt.subplots()
    x1 = np.linspace(1, len(fit), len(fit))
    y1 = np.array(fit)
    ax.plot(x1, y1)
    plt.show()
