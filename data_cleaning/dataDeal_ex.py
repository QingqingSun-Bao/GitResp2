import numpy as np
from sqlalchemy import create_engine
import pymysql
import pandas as pd

# 获取实验处理
engine=create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
data=pd.read_sql(str(2009),con=engine)
A=[]
df2=set(list(data.loc[:,'样地号']))
for ex in df2:
    df1=data[data['样地号']==str(float(ex))].index.tolist()
    A.append(data.loc[df1[0],['样地号','顺序','氮素','频率','刈割']])
pd.DataFrame(A).to_excel('C:/Users/97899/Desktop/实验处理_site.xls')
# df2=set(list(data.loc[:,'顺序']))
# for ex in range(1,39):
#     df1=data[data['顺序']==str(float(ex))].index.tolist()
#     A.append(data.loc[df1[0],['顺序','氮素','频率','刈割']])
# pd.DataFrame(A,columns=['顺序','氮素','频率','刈割']).to_excel('C:/Users/97899/Desktop/实验处理_ex.xls')