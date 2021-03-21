import pandas as pd
from scipy.stats import ttest_rel


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

'''T test about stability of biomass'''

path = 'C:/Users/97899/Desktop/N/'
df_bio = pd.read_excel(path + 'biomass.xls')
df_site = pd.read_excel(path + '实验处理_site.xls')
columns = ['顺序', '氮素', '频率', '刈割']
df_bio = pd.concat([df_bio, pd.DataFrame(columns=columns)])

'''biomass and experiment deal'''

for item in range(df_site.shape[0]):
    for jtem in range(df_site.shape[0]):
        if int(df_bio.iloc[item, 0]) == int(df_site.iloc[jtem, 1]):
            df_bio.loc[item, '顺序'] = df_site.iloc[jtem, 2]
            df_bio.loc[item, '氮素'] = df_site.iloc[jtem, 3]
            df_bio.loc[item, '频率'] = df_site.iloc[jtem, 4]
            df_bio.loc[item, '刈割'] = df_site.iloc[jtem, 5]
gb = df_bio.groupby('顺序')
'''配对样本T检验'''
D = {}
D_p = {}
s=0
m1=0
n1=0
writer = pd.ExcelWriter(path + 'Ttest_biomass.xls')
for g in gb:
    t_dict = {}
    p_dict = {}
    for xtem in range(2008, 2020):
        m1+=1
        t_p = ttest_rel(g[1][xtem], g[1][xtem + 1])
        t_dict[xtem] = t_p[0]
        p_dict[xtem] = t_p[1]
        if t_p[1] > 0.05:
            n1+= 1
    print(n1/m1)
    s+=n1/m1
    D[g[0]] = t_dict
    D_p[g[0]] = p_dict
pd.DataFrame(D).T.to_excel(writer,sheet_name='T_statistic')
pd.DataFrame(D_p).T.to_excel(writer,sheet_name='P_values')
writer.save()
writer.close()
print('稳定数据的占比', s/38)
print(pd.DataFrame(D).T)

