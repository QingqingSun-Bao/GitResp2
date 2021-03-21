# -*- coding: utf-8 -*-
# @Time:2021/2/213:01
# @File:Classfier.py
# @Software:PyCharm

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


# def all_bio_zhu(data, ex, zuhe, spec):
#     bio = 0
#     zhu = 0
#     data = data[data["顺序"] == str(float(ex))]
#     for zh in zuhe:
#         dt1 = data[data["样地号"] == zh]
#         bio_ = 0
#         zhu_ = 0
#         # 某块样地的总生物量
#         for item in spec:
#             bio_p = dt1[dt1["物种"] == item]["干重g"]
#             bio_ = bio_ + float(bio_p)
#             # 某块样地的总株丛数
#             zhu_p = dt1[dt1["物种"] == item]["株丛数"].values
#             for j in zhu_p:
#                 if j is not None:
#                         zhu_ = zhu_ + float(j)
#
#         bio = bio + bio_
#         zhu = zhu + zhu_
#     avg_bio = bio / len(zuhe)
#     avg_zhu = zhu / len(zuhe)
#     return avg_bio, avg_zhu


def all_bio_zhu_no(data, ex):
    bio = 0
    zhu = 0
    data = data[data["顺序"] == str(float(ex))]
    site=set(data["样地号"].tolist())
    # print("样地号",site)
    # 所有样地的总生物量
    for item in site:
        bio_ = 0
        zhu_ = 0
        dt1=data[data["样地号"]==item]["干重g"]
        dt2=data[data["样地号"]==item]["株丛数"].values
        for jtem in dt1:
            bio_ = bio_ + float(jtem)
        # 某块样地的总株丛数
        for xtem in dt2:
            if xtem is not None:
                zhu_ = zhu_ + float(xtem)
        bio = bio + bio_
        zhu = zhu + zhu_
    avg_bio = bio / 10
    avg_zhu = zhu / 10
    return avg_bio, avg_zhu

# def all_bio_zhu_no(data, ex, spec):
#     bio = []
#     zhu = []
#     data = data[data["顺序"] == str(float(ex))]
#     site = set(data["样地号"].tolist())
#     # print("样地号",site)
#     # 所有样地的总生物量
#     for item in site:
#         dt=data[data["样地号"] == item]
#         spe=set(dt["物种"].tolist())
#         if spec in spe:
#             dt1 = dt[dt["物种"]==spec]["干重g"].values
#             dt2 = dt[dt["物种"]==spec]["株丛数"].values
#             bio.append(float(dt1))
#             if dt2[0] is not None:
#                 zhu.append(float(dt2))
#     avg_bio = np.mean(bio)
#     avg_zhu = np.mean(zhu)
#     return avg_bio, avg_zhu


def main():
    path = "C:/Users/97899/Desktop/N/"
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot20.txt")
    zuhe = LoadDict(path + "Zuhe/Zuhe_20.txt")
    deal_ex = pd.read_excel(path + "实验处理_ex.xls")
    deal_ex.set_index(["顺序"], inplace=True)
    temp = pd.read_excel(path + "Enveriment/weather_temp.xls")
    temp.set_index(['Unnamed: 0'], inplace=True)
    rain = pd.read_excel(path + "Enveriment/weather_rain.xls")
    rain.set_index(['Unnamed: 0'], inplace=True)
    # print(deal_ex)
    ring = {}
    chain = {}
    no_comp = {}
    ring["bio"], ring["zhu"], ring["N"], ring["rain"], ring["temp"], ring["M"], ring["F"], ring[
        "Year"] = [], [], [], [], [], [], [], []
    chain["bio"], chain["zhu"], chain["N"], chain["rain"], chain["temp"], chain["M"], chain[
        "F"], chain["Year"] = [], [], [], [], [], [], [], []
    no_comp["bio"], no_comp["zhu"], no_comp["N"], no_comp["rain"], no_comp["temp"], no_comp["M"], no_comp[
        "F"], no_comp["Year"] = [], [], [], [], [], [], [], []

    # spec="糙隐子草"

    for year in range(2008, 2020):
        ring_chain = pd.read_excel(path + "Network/circle21.xls", sheet_name=str(year))
        ring_chain.set_index(['Unnamed: 0'], inplace=True)
        bio_root = pd.read_sql(str(year), con=engine)
        print(year)
        temp1 = temp.loc[year, 0]
        rain1 = rain.loc[year, 0]
        for ex in range(1, 39):
            if ring_chain.loc[ex, 3] >= 0:
                # print(ex)
                if ring_chain.loc[ex, 3] == 0:
                    bio, zhu = all_bio_zhu_no(bio_root, ex)
                    chain["bio"].append(bio)
                    chain["zhu"].append(zhu)
                    chain["N"].append(deal_ex.loc[ex, "氮素"])
                    chain["rain"].append(rain1)
                    chain["temp"].append(temp1)
                    chain["M"].append(deal_ex.loc[ex, "刈割"])
                    chain["F"].append(deal_ex.loc[ex, "频率"])
                    chain["Year"].append(year)
                else:
                    bio, zhu = all_bio_zhu_no(bio_root, ex)
                    ring["bio"].append(bio)
                    ring["zhu"].append(zhu)
                    ring["N"].append(deal_ex.loc[ex, "氮素"])
                    ring["rain"].append(rain1)
                    ring["temp"].append(temp1)
                    ring["M"].append(deal_ex.loc[ex, "刈割"])
                    ring["F"].append(deal_ex.loc[ex, "频率"])
                    ring["Year"].append(year)
            else:
                bio, zhu = all_bio_zhu_no(bio_root, ex)
                no_comp["bio"].append(bio)
                no_comp["zhu"].append(zhu)
                no_comp["N"].append(deal_ex.loc[ex, "氮素"])
                no_comp["rain"].append(rain1)
                no_comp["temp"].append(temp1)
                no_comp["M"].append(deal_ex.loc[ex, "刈割"])
                no_comp["F"].append(deal_ex.loc[ex, "频率"])
                no_comp["Year"].append(year)

    RoC = ["Loop"] * len(ring["M"]) + ["chain"] * len(chain["M"]) + ["non_competitive"] * len(no_comp["M"])
    # C = {"type": RoC, "Year": ring["Year"] + chain["Year"] + no_comp["Year"],
    #      "zhu": ring["zhu"] + chain["zhu"] + no_comp["zhu"],
    #      "bio": ring["bio"] + chain["bio"] + no_comp["bio"], "N": ring["N"] + chain["N"] + no_comp["N"],
    #      "F": ring["F"] + chain["F"] + no_comp["F"],
    #      "M": ring["M"] + chain["M"] + no_comp["M"], "rain": ring["rain"] + chain["rain"] + no_comp["rain"],
    #      "temp": ring["temp"] + chain["temp"] + no_comp["temp"]}
    RoC = ["Loop"] * len(ring["M"]) + ["chain"] * len(chain["M"])
    C = {"type": RoC, "Year": ring["Year"] + chain["Year"],
         "zhu": ring["zhu"] + chain["zhu"],
         "bio": ring["bio"] + chain["bio"], "N": ring["N"] + chain["N"],
         "F": ring["F"] + chain["F"],
         "M": ring["M"] + chain["M"], "rain": ring["rain"] + chain["rain"],
         "temp": ring["temp"] + chain["temp"]}

    data = pd.DataFrame(C)
    data.to_excel(path + "Classifier/data_no.xls")


main()
