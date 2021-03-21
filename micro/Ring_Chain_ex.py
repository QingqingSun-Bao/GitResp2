"""物种组合以环式或者链式出现的实验ex"""

'''导入矩阵字典'''


def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''保存文件'''


def savedict(datadict, name):
    file = open("C:/Users/97899/Desktop/N/Zuhe/" + name + ".txt", "w", encoding='utf8')
    file.write(str(datadict))
    file.close()


def main():
    path = "C:/Users/97899/Desktop/N/Zuhe/"
    # 所有试验样地的物种组合
    Zuhe = LoadDict(path + "Zuhe_20.txt")
    # 导入各物种组合
    dou_zh = LoadDict(path + "two_or21.txt")
    sin_zh = LoadDict(path + "single_or21.txt")
    th_zh = LoadDict(path + "th_or21.txt")
    # 构建物种组合以环式或链式出现的实验ex
    si_ex = {}
    do_ex = {}
    th_ex = {}
    for key1 in sin_zh:
        si_ex[key1[0]]=[]
        for year in range(2008, 2021):
            for ex in range(1, 39):
                if key1[0] in set(Zuhe[year][ex]):
                    # print("存在")
                    si_ex[key1[0]].append([year, ex])
    for key2 in dou_zh:
        do_ex[tuple(key2[0])]=[]
        for year in range(2008, 2021):
            for ex in range(1, 39):
                if set(key2[0]) & set(Zuhe[year][ex]) == set(key2[0]):
                    do_ex[tuple(key2[0])].append([year, ex])
    for key3 in th_zh:
        th_ex[tuple(key3[0])] = []
        for year in range(2008, 2021):
            for ex in range(1, 39):
                if set(key3[0]) & set(Zuhe[year][ex]) == set(key3[0]):
                    th_ex[tuple(key3[0])].append([year, ex])
    savedict(si_ex, "single_ex21")
    savedict(do_ex, "two_ex21")
    savedict(th_ex, "th_ex21")


main()
