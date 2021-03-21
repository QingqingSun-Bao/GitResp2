"""物种环数与降雨量"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

'''画图:降水量与环数统计'''

pd.set_option('display.max_columns', 100)


def raincircle(df, Weather0, Weather1):
    fig, ((ax0), (ax1)) = plt.subplots(nrows=2, ncols=1)
    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = 'SimHei'
    new_ticks = np.linspace(1, 11, 11)
    x = np.linspace(2008, 2018, 11)
    labels = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2017', '2018', '2019']
    # 年降水量与生长季降水量
    ax0.plot(x, Weather0)
    ax0.plot(x, Weather1)
    ax0.legend(('年降水量', '生长季降水量'), loc='upper right', shadow=True)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_title('降雨量年际变化趋势')
    # 环数柱状图
    x = np.arange(len(labels))
    width = 0.15
    Three = list(df.loc[3, :])
    Four = df.loc[4, :]
    Five = df.loc[5, :]
    Six = df.loc[6, :]
    ax1.bar(x - width - 0.15, Three, width, label='3物种环')
    ax1.bar(x - width / 2 - 0.05, Four, width, label='4物种环')
    ax1.bar(x + width / 2 + 0, Five, width, label='5物种环')
    ax1.bar(x + width + 0.05, Six, width, label='6物种环')
    # ax1.set_ylabel('Scores')
    # ax1.set_title('Scores by group and gender')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('自然状态下非传递性环的变化趋势')
    ax1.legend()

    plt.show()


'''拟合环数与降雨量的关系'''


def Linearfigure(df, Weather0, Weather1):
    # fg, ((ax3, ax4), (ax5)) = plt.subplots(nrows=2, ncols=2)
    x3 = Weather1
    y3 = df.iloc[0, 0:7]
    plt.scatter(x3, y3)
    X_train, X_test, Y_train, Y_test = train_test_split(x3, y3, train_size=0.9)
    model = LinearRegression()
    X_train = X_train.reshape(-1, 1)
    model.fit(X_train, Y_train)
    a = model.intercept_  # 截距
    b = model.coef_  # 回归系数
    R_squre = get_lr_stats(X_train, Y_train, model)
    print(R_squre)
    # 训练数据的预测值
    y_train_pred = model.predict(X_train)
    # 绘制最佳拟合线：标签用的是训练数据集中的极值预测值
    X_train_pred = [min(X_train), max(X_train)]
    y_train_pred = [a + b * min(X_train), a + b * max(X_train)]
    plt.plot(X_train_pred, y_train_pred, color='green', linewidth=3, label="best line")

    plt.show()


'''获得回归的拟合值'''


def get_lr_stats(x, y, model):
    # message0 = '一元线性回归方程为: ' + '\ty' + '=' + str(model.intercept_) + ' + ' + str(model.coef_[0]) + '*x'
    y_prd = model.predict(x)
    # Regression = sum((y_prd - np.mean(y))**2) # 回归平方和
    Residual = sum((y - y_prd) ** 2)  # 残差平方和
    total = sum((y - np.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total  # 相关性系数R^2
    # message1 = ('相关系数(R^2)： ' + str(R_square) + '；' + '\n' + '总体平方和(TSS)： ' + str(total) + '；' + '\n')
    # message2 = ('回归平方和(RSS)： ' + str(Regression) + '；' + '\n残差平方和(ESS)： ' +  str(Residual) + '；' + '\n')
    return R_square


def quaradic(x, df):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = 'SimHei'
    for i in range(df.shape[1]):
        y = df.iloc[i, :7]
        print(y)
        f = np.polyfit(x, y, 2)
        p = np.poly1d(f)
        print('p1 is :\n', p)
        plt.scatter(x, y)
        x.tolist()
        x.sort()
        x = np.array(x)
        y_values = p(x)
        y = np.array(y)
        plt.plot(x, y_values, 'r')
        plt.title('非传递性网络' + str(i + 3) + '物种环数与降水量的关系')
        plt.show()


def main():
    path = 'C:/Users/97899/Desktop/'
    f = open(path + 'Nature_year/' + 'weather.txt', 'r+')
    dict = eval(f.read())
    f.close()
    Weather0 = []
    Weather1 = []
    for key in dict.keys():
        Weather0.append(float(dict[key][1]))
        Weather1.append(float(dict[key][0]))
    Weather0 = np.array(Weather0)
    Weather1 = np.array(Weather1[0:7])
    print(Weather1)
    df = pd.read_excel(path + '非传递性环的数量.xls', sheet_name='23')
    # df.columns=['环数','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
    df = df.set_index(['环数'])
    # raincircle(df, Weather0, Weather1)
    # Linearfigure(df,Weather0,Weather1)
    quaradic(Weather1, df)


main()
