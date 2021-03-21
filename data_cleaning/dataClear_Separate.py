import pandas as pd

df=pd.read_excel('C:/Users/97899/Desktop/氮沉降及功能群数据.xls',sheet_name='2008-2020连续施肥的结果')
D={}
df_g=df.groupby('年份')
# 按年份分组存入表格
writer=pd.ExcelWriter("C:/Users/97899/Desktop/内蒙古氮沉降数据.xls")
for g in df_g:
   g[1].to_excel(writer, sheet_name=str(g[1].iloc[2,1]))
writer.save()
writer.close()
