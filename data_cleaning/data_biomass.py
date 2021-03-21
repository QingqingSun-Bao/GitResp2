import pandas as pd
import pymysql
from sqlalchemy import create_engine
import numpy as np

pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
# engine=create_engine("mysql+pymysql://root:root@localhost:3306/nchenjiang?charset=utf8")
engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')

'''每年各个样地中的生物量'''


# D_bio={}
#
# for i in range(2008,2021):
#     D_bio[i]={}
#     data=pd.read_sql(str(i),con=engine)
#     data_N=data.loc[:,['样地号','干重g']]
#     dp=data_N.groupby(['样地号'])
#     print('第%d年'%(i))
#     for item in dp:
#         D_bio[i][item[0]]=sum([float(j) for j in item[1]['干重g']])
# df_bio=pd.DataFrame(D_bio)
# # print(df_bio)
# df_bio.to_excel('C:/Users/97899/Desktop/N/Biomass/biomass.xls',sheet_name="bio_site")


def Loaddic(path):
    fr = open(path, encoding='UTF-8')
    # 'unicode_escape'
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


"""各物种组配下的生物量"""
df_bio = pd.read_excel('C:/Users/97899/Desktop/N/Biomass/biomass.xls', sheet_name="bio_site")
Zuhe_bio = Loaddic("C:/Users/97899/Desktop/N/Zuhe/Zuhe_site.txt")
df_bio.set_index(["site"], inplace=True)
print(df_bio)
ex_bio = {}
ind = np.linspace(2009, 2019, 11).tolist()
for item in ind:
    ex_bio[item] = []
    for jtem in Zuhe_bio[item]:
        if jtem != -0.15:
            sum = 0
            for i in jtem:
                sum = sum + df_bio.loc[int(i), item]
            ex_bio[item].append(sum / len(jtem))
        else:
            ex_bio[item].append(-0.15)

pd.DataFrame(ex_bio).to_excel('C:/Users/97899/Desktop/N/Biomass/bio_ex.xls')
