import pandas as pd
import numpy as np
from sqlalchemy import create_engine

'''single biomass'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic




def main():
    path = "C:/Users/97899/Desktop/N/"
    single_ex = LoadDict(path + "Zuhe/single_ex20.txt")
    single_nested_ex = LoadDict(path + "Zuhe/single_nested_ex21.txt")
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    spec = "砂韭"
    # 黄囊苔草、大针茅、羊草、糙隐子草、羽茅
    hu_nested = {}
    bio_hu = {}
    hu_no_nested = {}
    print(spec)
    for year in range(2014, 2015):
        hu_nested[year] = {}
        bio_hu[year] = {}
        hu_no_nested[year] = {}
        for ex in range(31, 32):
            print('所在年份%s,第%s个处理' % (year, ex))
            print("样地号", zuhe_plot[year][ex])
            if [year, ex] in single_nested_ex[spec]:
               print('所在年份%s,第%s个处理'%(year,ex))
               print("样地号",zuhe_plot[year][ex])


main()
