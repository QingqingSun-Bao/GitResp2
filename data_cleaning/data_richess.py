import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:root@localhost:3306/nchenjiang?charset=utf8')

"""各处理下出现的总物种数"""
# D = {}
# site_rich = {}
# # A 总共出现过的物种数
# A = set()
# for year in range(2008, 2021):
#     df = pd.read_sql(str(year), con=engine)
#     ex_rich = []
#     for ex in range(1, 39):
#         df_site = df[df['顺序'] == str(float(ex))]
#         gb_site = df_site.groupby('样地号')
#         all_rich = set()
#         for jtem in gb_site:
#             all_rich = all_rich | set(jtem[1]['物种'])
#         A = A | all_rich
#         ex_rich.append(len(all_rich))
#     site_rich[str(year)] = ex_rich
# print(pd.DataFrame(site_rich))
# pd.DataFrame(site_rich).to_excel('C:/Users/97899/Desktop/N/Richness/rich_site.xls', sheet_name="all_rich")
# print('出现过得物种数', len(A))


def savedict(datadict):
    file = open("C:/Users/97899/Desktop/N/Zuhe/Zuhe_site.txt", "w")
    file.write(str(datadict))
    file.close()


def Loaddic(path):
    fr = open(path, encoding='UTF-8')
    # 'unicode_escape'
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


for year in range(2008, 2020):
    df = pd.read_sql(str(year), con=engine)
    gb = df.groupby("顺序")
    for g in gb:
        A = set(g[1]["物种"])

'''组配所出现样地中alpha and beta diversity'''
alpha_rich = {}  # 样地的平均物种数
gamma_rich = {}  # 处理下的所有物种数
beta_rich = {}
Zuhe = Loaddic("C:/Users/97899/Desktop/N/Zuhe/Zuhe_20.txt")
site = {}
lose_seed = {}  # 丧失的物种
new_seed = {}  # 新物种的植入
"""新物种的植入与旧物种的丧失"""
# 08年各个处理下的物种
df_nature = pd.read_sql(str(2008), con=engine)
gn = df_nature.groupby("顺序")
nat_seed = []
for g1 in gn:
    A = set(g1[1]["物种"])
    nat_seed.append(A)
for year in range(2008, 2021):
    # 取某年的最优组合
    df = pd.read_sql(str(year), con=engine)
    total_rich = []
    avg_rich = []
    turnover = []
    site[year] = []
    lose = []
    new = []
    for ex in range(1, 39):
        # 某年的某个样地的组合
        ass = Zuhe[year][ex]
        num_site = []
        A = set()
        if len(ass) == 0:
            # print(ex,"没有最优组合")
            avg_rich.append(-0.15)
            total_rich.append(-0.15)
            turnover.append(-0.15)
            lose.append(-0.15)
            new.append(-0.15)
            site[year].append(-0.15)
        else:
            df_site = df[df['顺序'] == str(float(ex))]
            gb_site = df_site.groupby('样地号')
            site_num = []
            for jtem in gb_site:
                # 取出现过最优组配的样地jtem
                if set(ass) & set(jtem[1]['物种']) == set(ass):
                    # 各个样地的总物种
                    site_num.append(int(float(jtem[0])))
                    A = A | set(jtem[1]['物种'])
                    # 各地物种数量
                    num_site.append(len(jtem[1]['物种']))
            site[year].append(site_num)
            # 计算丧失的物种数以及新植入的物种数
            maintain = nat_seed[ex - 1] & A
            lose.append(len(nat_seed[ex - 1]) - len(maintain))
            new.append(len(A) - len(maintain))
            avg_rich.append(np.mean(num_site))
            # 小尺度1平方米的平均物种丰富度
            total_rich.append(len(A))
            # gamma 样地中出现的所有物种数
            turnover.append(1 - np.mean(num_site) / len(A))
    # print(year, len(avg_rich), len(total_rich))
    print("总样地数", len(site))
    gamma_rich[float(year)] = total_rich
    alpha_rich[float(year)] = avg_rich
    beta_rich[float(year)] = turnover
    lose_seed[float(year)] = lose
    new_seed[float(year)] = new
# savedict(site)
write = pd.ExcelWriter("C:/Users/97899/Desktop/N/Richness/rich_null21.xls")
pd.DataFrame(gamma_rich).to_excel(write, sheet_name="gamma")
pd.DataFrame(alpha_rich).to_excel(write, sheet_name="alpha")
pd.DataFrame(beta_rich).to_excel(write, sheet_name="beta")
pd.DataFrame(lose_seed).to_excel(write, sheet_name="lose")
pd.DataFrame(new_seed).to_excel(write, sheet_name="new")
write.save()
write.close()
