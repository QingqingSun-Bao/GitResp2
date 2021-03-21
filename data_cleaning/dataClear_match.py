import pandas as pd

path = 'C:/Users/97899/Desktop/'
df1 = pd.read_excel(path + 'N2020.xls', sheet_name='2020植物数据')
df2 = pd.read_excel(path + '实验处理.xls', sheet_name='Sheet1')
columns = ['顺序', '氮素', '频率', '刈割']
df = pd.concat([df1, pd.DataFrame(columns=columns)])
# df=df1.reindex(columns=)
print(df.columns)
for i in range(df.shape[0]):
    for j in range(df2.shape[0]):
        if df.iloc[i, 2] == df2.iloc[j, 1]:
            df.iloc[i, 12] = df2.iloc[j, 2]
            df.iloc[i, 13] = df2.iloc[j, 3]
            df.iloc[i, 14] = df2.iloc[j, 4]
            df.iloc[i, 15] = df2.iloc[j, 5]
gb=df.groupby(['C/T'])
for item in gb:
    item[1].to_excel(path +str(item[0])+'N_2020.xls')
