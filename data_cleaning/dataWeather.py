import pandas as pd
from sqlalchemy import create_engine
import pymysql
import numpy as np



def temp(engine,monthset):
    D={}
    for item in range(2008, 2021):
        if item in range(2008, 2018):
            col = '自动气象站日平均气温/℃'
        else:
            col = '日平均气温/℃'
        df = pd.read_sql(str(item), con=engine)
        # df_1 = df[df['人工监测日降水量/mm'].notna()]
        df_1 = df[df[col].notna()]
        df_1 = df_1.fillna(0)
        sum_1 = 0
        D[item] = []
        '''生长季平均气温'''
        sum_all = 0
        for j in monthset:
            df_2 = df_1[df_1['月'] == str(j)]
            days = 0
            sum_0=0
            for i in df_2[col]:
                days += 1
                if len(i) == 1:
                    i = 0.0
                sum_0 = sum_0 + float(i)
            sum_all = sum_all + sum_0 / days
        sum_all = sum_all / len(monthset)
        D[item].append('%.2f' % sum_all)
        '''全年日均温度'''
        days3=0
        for i in df_1[col]:
            if len(i) == 1:
                i = 0.0
            days3+=1
            sum_1 = sum_1 + float(i)
        D[item].append('%.2f' % (sum_1/days3))
        '''1-7月份日平均温度'''
        sum_all_7 = 0
        for j in range(1,8):
            df_2 = df_1[df_1['月'] == str(float(j))]
            days = 0
            sum_0 = 0
            for i in df_2[col]:
                days += 1
                if len(i) == 1:
                    i = 0.0
                sum_0 = sum_0 + float(i)
            sum_all_7 = sum_all_7 + sum_0 / days
        sum_all_7 = sum_all_7 / len(monthset)
        D[item].append('%.2f' % sum_all_7)



    return D

'''降水量'''
def fall(engine,monthset):
    D={}
    for item in range(2008,2021):
        print(item)
        col = '人工监测日降水量/mm'
        df = pd.read_sql(str(item), con=engine)
        df_1 = df.fillna(0)

        D[item] = []
        sum_1 = 0
        sum_0 = 0
        n0=0
        '''生长季降水量'''
        for j in monthset:
            df_2 = df_1[df_1['月'] == str(j)]
            n0=n0+df_2.shape[0]

            for i in df_2[col]:
                # print(j,i)
                if type(i)==str:
                    if len(i)==1:
                        i=0.0
                    else:
                        float(i)
                sum_0 = sum_0 + float(i)
        # 生长季降雨量的总和
        print(sum_0)
        D[item].append('%.2f' % sum_0)
        # 生长季降雨量每天
        D[item].append('%.2f' % (float(sum_0)/n0))

        '''全年降水量'''

        n1=np.shape(df_1)[0]
        for i in df_1[col]:
            if type(i) == str:
                if len(i) == 1:
                    i = 0.0
                else:
                    float(i)
            sum_1 = sum_1 + float(i)
        D[item].append('%.2f' % sum_1)
        D[item].append('%.2f' % (sum_1 / n1))
        '''1-7月降水量'''
        sum_2=0
        n2=0
        for j in range(5,7):
            df_3 = df_1[df_1['月'] == str(float(j))]
            n2=n2+np.shape(df_3)[0]
            for i in df_3[col]:
                if type(i) == str:
                    if len(i) == 1:
                        i = 0.0
                    else:
                        float(i)
                sum_2 = sum_2 + float(i)
        D[item].append('%.2f' % sum_2)
        D[item].append('%.2f' % (sum_2 / n2))
        '''9-12月降水量'''
        sum_3 = 0
        n3 = 0
        for j in range(9,13):
            df_3 = df_1[df_1['月'] == str(float(j))]
            n3 = n3 + np.shape(df_3)[0]
            for i in df_3[col]:
                if type(i) == str:
                    if len(i) == 1:
                        i = 0.0
                    else:
                        float(i)
                sum_3 = sum_3 + float(i)
        D[item].append('%.2f' % sum_3)
        D[item].append('%.2f' % (sum_3 / n3))
    return D


def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/weather?charset=utf8')
    monthset = [5.0, 6.0, 7.0, 8.0]
    path = "C:/Users/97899/Desktop/N/Enveriment/"
    # 获取降雨量
    D = fall(engine,monthset)
    rain=pd.DataFrame(D).T
    rain.columns=["season","avg_sea","all","avg_all","7","avg_7","9","avg_9"]
    # 21为全的2020
    rain.to_excel(path + 'weather_rain_21.xls')
    # 获取温度
    # D1 = temp(engine,monthset)
    # pd.DataFrame(D1).T.to_excel(path + 'weather_temp_21.xls')
    # "0-season,1-all,2-1~7"

main()