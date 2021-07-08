# -*- coding: utf-8 -*-
# @Time:2021/6/2512:48
# @File:collfile.py
def cal_similar(dict1, dict2):
    result_similar = {}
    for vs in dict1.values():
        print("vs", vs)
        for v1 in vs:
            if v1 not in result_similar:
                result_similar[v1] = {}
            for v2 in vs:
                if v1 == v2:
                    continue
                c = len(set(dict2.get(v1, []) + dict2.get(v2, [])))
                result_similar[v1][v2] = result_similar[v1].get(v2, 0) + 1.0 / c
    result_similar = dict([[key, value.items()] for [key, value] in result_similar.items()])
    return result_similar
cal_similar({},{})
