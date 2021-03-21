import pandas as pd
import numpy as np
from sqlalchemy import create_engine

'''single biomass'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def spe_bio(data, ex, zuhe,spec):
    zhu=0
    data = data[data["顺序"] == str(float(ex))]
    for zh in zuhe:
        dt1 = data[data["样地号"] == zh]
        dt3 = dt1[dt1["物种"] == spec]["株丛数"]
        zhu = zhu+float(dt3)

    return zhu/len(zuhe)


def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    path = "C:/Users/97899/Desktop/N/"
    single_ex = LoadDict(path + "Zuhe/single_ex20.txt")
    single_nested_ex = LoadDict(path + "Zuhe/single_nested_ex20.txt")
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    spec = "羽茅"
    # 黄囊苔草、大针茅、羊草、糙隐子草、羽茅
    hu_nested = {}
    bio_hu = {}
    hu_no_nested = {}
    for year in range(2008, 2021):
        hu_nested[year] = {}
        bio_hu[year] = {}
        hu_no_nested[year] = {}
        df_bio = pd.read_sql(str(year), con=engine)
        for ex in range(1, 39):
            print(year)
            # nest
            if [year, ex] in single_nested_ex[spec]:
                hu_nested[year][ex] = spe_bio(df_bio, ex, zuhe_plot[year][ex],spec)
            else:
                hu_nested[year][ex] = 0
                #all
            if [year, ex] in single_ex[spec]:
                bio_hu[year][ex] = spe_bio(df_bio, ex, zuhe_plot[year][ex],spec)
            else:
                bio_hu[year][ex] = 0
            # no nest
            # 存在物种，但物种不以环的形式存在
            if ([year, ex] in single_ex[spec]) and ([year, ex] not in single_nested_ex[spec]):
                hu_no_nested[year][ex] = spe_bio(df_bio, ex, zuhe_plot[year][ex],spec)
                print("存在")
            else:
                hu_no_nested[year][ex] = 0
    write = pd.ExcelWriter(path + "Single/yu/yu_rem.xls")
    pd.DataFrame(hu_nested).to_excel(write, sheet_name="yu_loop_rem")
    pd.DataFrame(bio_hu).to_excel(write, sheet_name="yu_rem")
    pd.DataFrame(hu_no_nested).to_excel(write,sheet_name="yu_chain_rem")
    write.save()
    write.close()


main()
