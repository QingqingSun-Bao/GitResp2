import pandas as pd
import numpy as np
from sqlalchemy import create_engine

path='C:/Users/97899/Desktop/Biomass_apriori/'

'''取出自然状态的物种组合'''
def Natureconponet(year):
    f=open(path+'Assemb'+str(year)+'-'+str(0)+'.txt',encoding='uft-8')
    Dic=eval(f.read())  # 将str转化为dict
    f.close()
    return Dic




def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    year=float(2009)
    '''取出自然状态下的组合'''
    Con_specise=Natureconponet(year)
    for ex in range(2,39):
        '''物种在各个地点的丰度分布'''




main()
