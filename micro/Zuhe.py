import pandas as pd
import numpy as np
from Load_Save import LoadDict



# 计算物种出现的频率
def FreSpec(Zuhe, ex):
    Ind = np.linspace(2008, 2020, 13).tolist()
    # Ind = [2008]
    D_spec = {}
    for year in Ind:
        for seed in Zuhe[year][ex]:
            if seed not in D_spec.keys():
                D_spec[seed] = 1
            else:
                D_spec[seed] += 1
    D_spec_order=sorted(D_spec.items(),key=lambda x:x[1],reverse=True)
    return D_spec_order


def main():
    path = "C:/Users/97899/Desktop/N/Zuhe/Zuhe_20.txt"
    Zuhe = LoadDict(path)
    D_ex={}
    for ex in range(1,39):
        D_ex[ex]=FreSpec(Zuhe, ex)
    print(D_ex)
main()
