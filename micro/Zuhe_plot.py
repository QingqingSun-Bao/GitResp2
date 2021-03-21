import pandas as pd
import numpy as np
from sqlalchemy import create_engine

"""找出所有三物种组合所在的ex"""
'''导入矩阵字典'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''保存文件'''


def savedict(datadict):
    file = open("C:/Users/97899/Desktop/N/Zuhe/Zuhe_plot20.txt", "w",encoding='utf8')
    file.write(str(datadict))
    file.close()


'''找到样地'''


def zuhe_plot(zuhe, ex, df):
    plot = []
    # 取出实验样地
    df = df[df["顺序"] == str(float(ex))]
    # 根据样地号分组
    gb = df.groupby("样地号")
    for g in gb:
        if set(zuhe) & set(g[1]["物种"]) == set(zuhe):
            plot.append(g[0])
    return plot


def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    path = "C:/Users/97899/Desktop/N/Zuhe/"
    # 所有试验样地的物种组合
    Zuhe=LoadDict(path+"Zuhe_20.txt")
    # 物种组合出现的样地
    stt_plot = {}
    for year in range(2008, 2021):
        stt_plot[year]={}
        df=pd.read_sql(str(year),con=engine)
        for ex in range(1, 39):
            stt_plot[year][ex]=zuhe_plot(Zuhe[year][ex],ex,df)
    savedict(stt_plot)
    print(stt_plot)


main()
