import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def Loaddic(path):
    fr = open(path, encoding='UTF-8')
    # 'unicode_escape'
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def spe_bio(data, zuhe_plot, spec):
    bio = 0
    n=0
    for zh in zuhe_plot:
        dt1 = data[data["样地号"] == zh]
        dt2 = dt1[dt1["物种"] == spec]["干重g"]
        dt3=dt1[dt1["物种"]==spec]["株丛数"].values
        print(zh)
        if dt3 is None:
            continue
        else:
            if float(dt3)==0:
                n=n+1
            else:
                bio = bio + float(dt2)
    return bio

def spe_zhu(data, zuhe_plot, spec):
    n=0
    zhu=0
    for zh in zuhe_plot:
        dt1 = data[data["样地号"] == zh]
        dt3=dt1[dt1["物种"]==spec]["株丛数"].values
        print(zh)
        if dt3 is None:
            continue
        else:
            if float(dt3)==0:
                n=n+1
            else:
                zhu=zhu+float(dt3)
    return zhu

def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    path = "C:/Users/97899/Desktop/N/"
    Zuhe = Loaddic(path + "Zuhe/Zuhe_20.txt")
    zuhe_plot = Loaddic(path + "Zuhe/Zuhe_plot20.txt")
    th_or=Loaddic(path + "Zuhe/th_or.txt")
    two_or=Loaddic(path + "Zuhe/two_or.txt")
    si_or=Loaddic(path + "Zuhe/single_or.txt")
    Bio = {}
    zhu={}
    for year in range(2008, 2021):
        df = pd.read_sql(str(year), con=engine)
        for ex in range(1, 39):
            data = df[df["顺序"] == str(float(ex))]
            D_lst = Bio.keys()
            # zhu_lst=zhu.keys()
            for spec in Zuhe[year][ex]:
                print(year,ex,spec)
                if str(spec) in D_lst:
                    Bio[str(spec)]= Bio[spec] + spe_bio(data, zuhe_plot[year][ex], str(spec))
                    zhu[str(spec)] = zhu[spec] + spe_zhu(data, zuhe_plot[year][ex], str(spec))
                else:
                    Bio[str(spec)]= spe_bio(data, zuhe_plot[year][ex], str(spec))
                    zhu[str(spec)] =spe_zhu(data, zuhe_plot[year][ex], str(spec))
                    D_lst = Bio.keys()
                    # zhu_lst=zhu.keys()
    D_si_bio={}
    for key in Bio.keys():
        D_si_bio[key]=Bio[key]/zhu[key]
    print(sorted(Bio.items(), key=lambda x: x[1], reverse=True))
    print(sorted(D_si_bio.items(), key=lambda x: x[1], reverse=True))
    all_num_th=0
    all_num_to=0
    all_num_si = 0
    for th in th_or:
        all_num_th=all_num_th+th[1]
    print(len(th_or), all_num_th)
    for to in two_or:
        all_num_to=all_num_th+to[1]
    print(len(two_or), all_num_to)
    for si in si_or:
        all_num_si=all_num_th+si[1]
    print(len(si_or), all_num_si)



main()
