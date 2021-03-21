from itertools import combinations
'''内嵌物种组合中，单物种、两物种、三物种以环装形式出现的样地'''

# 导入组合
def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


def savedict(datadict, doc):
    file = open("C:/Users/97899/Desktop/N/Zuhe/" + str(doc) + ".txt", "w")
    file.write(str(datadict))
    file.close()


'''统计三物种组合出现的次数'''


def sta_fre_th(D_all, th_lst, th_num,th_plot,year,ex):
    D_lst = list(D_all.keys())
    for thtem in th_lst:
        if len(thtem) < 3:
            break
        else:
            th_num = th_num + 1
            if tuple(thtem) not in D_lst:
                n = 0
                for Dtem in D_lst:
                    # 再次判断集合是否存在
                    if set(thtem) == set(Dtem):
                        D_all[tuple(Dtem)] = D_all[tuple(Dtem)] + 1
                        th_plot[tuple(Dtem)].append([year, ex])
                    else:
                        n = n + 1
                if n == len(D_lst):
                    D_all[tuple(thtem)] = 1
                    th_plot[tuple(thtem)]=[[year,ex]]

            else:
                D_all[tuple(thtem)] = D_all[tuple(thtem)] + 1
                th_plot[tuple(thtem)].append([year, ex])
    return D_all, th_num,th_plot


"""三物种组合中两物种出现的频率"""


def sta_fre_se(D_se, th_lst,th_plot,year,ex):
    D_lst = list(D_se.keys())
    for thtem in th_lst:
        # 取出一个三物种组合的元素进行两两组合
        if len(thtem) < 3:
            break
        else:
            th = list(combinations(thtem, 2))
            for tem in th:
                if tuple(tem) not in D_lst:
                    # 若不在字典的键里，则与所有的匹配
                    n = 0
                    for Dtem in D_lst:
                        if set(tem) & set(Dtem) == set(tem):
                            # 因为顺序，组合存在
                            D_se[tuple(Dtem)] = D_se[tuple(Dtem)] + 1
                            th_plot[tuple(Dtem)].append([year,ex])
                            break
                        else:
                            n = n+1
                    if n == len(D_lst):
                        D_lst = list(D_se.keys())
                        # 对字典的键随时更新
                        D_se[tuple(tem)] = 1
                        th_plot[tuple(tem)] = [[year, ex]]

                else:
                    D_se[tuple(tem)] = D_se[tuple(tem)] + 1
                    th_plot[tuple(tem)].append([year, ex])

    return D_se,th_plot


"""三物种组合中单个物种出现的频率"""


def sta_fre_fir(D_fir, th_lst,th_plot,year,ex):
    D_lst = list(D_fir.keys())
    for thtem in th_lst:
        if len(thtem) < 3:
            break
        else:
            for th in thtem:
                if str(th) not in D_lst:
                    D_fir[str(th)] = 1
                    th_plot[str(th)]=[[year,ex]]
                else:
                    D_fir[str(th)] = D_fir[str(th)] + 1
                    if [year,ex] not in th_plot[str(th)]:
                       th_plot[str(th)].append([year,ex])
    return D_fir,th_plot


def main():
    path = "C:/Users/97899/Desktop/N/Zuhe/all_th_nest21.txt"
    th_nest = LoadDict(path)
    D_th = {};D_se = {};D_fir = {};D_plot={};D_se_plot={};D_th_plot={}
    th_num = 0
    for year in range(2008, 2021):
        for ex in range(1, 39):
            ex = float(ex)
            D_th, th_num,D_th_plot= sta_fre_th(D_th, th_nest[year][ex], th_num,D_th_plot,year,ex)
            D_se,D_se_plot = sta_fre_se(D_se, th_nest[year][ex],D_se_plot,year,ex)
            D_fir,D_plot = sta_fre_fir(D_fir, th_nest[year][ex],D_plot,year,ex)
    savedict(D_plot,"single_nested_ex21")
    savedict(D_se_plot, "two_nested_ex21")
    savedict(D_th_plot, "th_nested_ex21")


            # 单个物种出现的实验样地
    # print("三物种组合", sorted(D_th.items(), key=lambda x: x[1], reverse=True))
    savedict(sorted(D_th.items(), key=lambda x: x[1], reverse=True), "th_or21")
    # print("两个物种", sorted(D_se.items(), key=lambda x: x[1], reverse=True))
    savedict(sorted(D_se.items(), key=lambda x: x[1], reverse=True), "two_or21")
    # print("一个物种", sorted(D_fir.items(), key=lambda x: x[1], reverse=True))
    savedict(sorted(D_fir.items(), key=lambda x: x[1], reverse=True), "single_or21")
    # print("总三物种组合数", th_num)


main()
