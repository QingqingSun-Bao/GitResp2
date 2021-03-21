import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from sqlalchemy import create_engine

'''挖掘各处理下的频繁项集'''
pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_columns', 100)


def APRIoRI(engine, year, ex):
    data = pd.read_sql(str(year), con=engine)
    data_ex = data[data['顺序'] == str(float(ex))]
    data_site = set(data_ex['样地号'])
    D = {}
    Selsect_set = []
    Spec_set = []
    for i in data_site:
        Site_spec = list(set(data_ex[data_ex['样地号'] == i]['物种']))
        Spec_set.append(Site_spec)
    te = TransactionEncoder()
    # 进行one-hot编码，0-1
    te_array = te.fit(Spec_set).transform(Spec_set)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # 用apriori找出频繁项集
    freq = apriori(df, min_support=0.5, use_colnames=True)
    Max_len = 0
    n = 0
    # 找到0.5概率出现的三物种以上的组合
    D_zuhe = []
    # 找到出现的所有物种组合所出现的次数
    D_zuhe_number = {}
    for item2 in reversed(freq['itemsets']):
        if len(item2) > 2:
            Spec_lis = [i for i in item2]
            D_zuhe.append(Spec_lis)
        if len(item2) not in D_zuhe_number.keys():
            D_zuhe_number[len(item2)] = 1
        else:
            D_zuhe_number[len(item2)] += 1

    for item in reversed(freq['itemsets']):
        if len(item) >= Max_len:
            Max_len = len(item)
            Spec_lis = [i for i in item]
            D[n] = []
            Selsect_set.append(Spec_lis)
            D[n].append(Spec_lis)
            site_temp = []
            for i in range(len(Spec_set)):
                if (set(Spec_set[i]) | item) == set(Spec_set[i]):
                    site_temp.append(list(data_site)[i])
            D[n].append(site_temp)
            n = n + 1
        else:
            break
    print(D)
    # 返回各个物种组合Selsect_set，以及包含所在样子的字典D,返回0.5以上的所有组合数
    return D_zuhe_number, D_zuhe


def get_zh_fre(D):
    D_zh_fre = {}
    for iex in range(1, 39):
        print('第%d个处理' % iex)
        D_zh_fre[iex] = {}
        for year in range(2008, 2021):
            for item in D[year][iex]:
                if tuple(item) not in D_zh_fre[iex].keys():
                    D_zh_fre[iex][tuple(item)] = 1
                else:
                    D_zh_fre[iex][tuple(item)] += 1
        D_zh_fre[iex]=sorted(D_zh_fre[iex].items(),key=lambda x:x[1],reverse=True)
        print(D_zh_fre[iex])
        print('每个处理下时间上出现的组合',len(D_zh_fre[iex]))
    return D_zh_fre





def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    zuhe_specise_D = {}
    # writer = pd.ExcelWriter('C:/Users/97899/Desktop/Zuhe.xls')
    for year in range(2008, 2021):
        zuhe_specise = {}
        Zuhe_number_D = {}
        for ex in range(1, 39):
            D_zuhe_number, D_zuhe = APRIoRI(engine, year, ex)
            Zuhe_number_D[str(ex)] = D_zuhe_number
            zuhe_specise[ex] = D_zuhe
        print('第' + str(year) + '年')
        zuhe_specise_D[year] = zuhe_specise
        print(zuhe_specise_D[year])
    D_zh_fre=get_zh_fre(zuhe_specise_D)
    print(D_zh_fre)

    #     pd.DataFrame(Zuhe_number_D).T.to_excel(writer, sheet_name=str(year))
    # writer.save()
    # writer.close()


main()
