# -*- coding: utf-8 -*-
# @Time:2021/3/2316:23
# @File:F_all_plot.py
# @Software:PyCharm

"""样地中功能群的组成"""
import pandas as pd
import os

pd.set_option("display.max_rows", 100)

path = "C:/Users/97899/Desktop/Function/"

for year in range(2008, 2021):
    df = pd.read_excel(path + "N_richment.xls", sheet_name=str(year))
    plots = set(df["样地号"].values)
    with open(path + "F_plot/%d.txt" % year, "w+") as f:
        # 判断系统是否为空，写入第一行
        if os.path.getsize(path + "F_plot/%d.txt" % year) == 0:
            f.write("样地号\t\t功能群")
            f.write("\n")
        for item in plots:
            df_1 = df[df["样地号"] == item]
            fun = list(set(df_1["功能群"].values))
            f.write(str(item) + "\t\t" + str(fun))
            f.write("\n")

"""样地中原始的功能群"""
"""环链中的功能群、个数以及种类"""
"""功能群对应的竞争力"""
