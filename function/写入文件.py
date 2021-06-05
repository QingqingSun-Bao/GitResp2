# -*- coding: utf-8 -*-
# @Time:2021/4/2815:27
# @File:写入文件.py

from numpy import *


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


if __name__ == "__main__":
    path = "C:/Users/97899/Desktop/N/"
    lst = LoadDict(path + "Zuhe/th_or21.txt")
    print(lst)
    # with open(path+"3.txt","a") as f:
    #     for item in dic:
    #         f.write("{:20}\t{:40}\t{:60}\t{:80}".format(item[0][0],item[0][1],item[0][2],str(item[1])))
    #         f.write("\n")
    dic_={}
    for i in range(len(lst)):
        for j in range(i+1,len(lst)):
            print(set(lst[i][0]),set(lst[j][0]))
            if set(lst[i][0])==set(lst[j][0]):
               print("y有重复")
    print({'羊草', '冰草','猪毛菜'}=={'猪毛菜','羊草',  '冰草'})
    # with open(path+"3_429.txt","a") as f:
    #
    #
    #
    #         f.write("{:20}\t{:40}\t{:60}\t{:80}".format(item[0][0],item[0][1],item[0][2],str(item[1])))
    #         f.write("\n")
