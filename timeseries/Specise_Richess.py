import numpy as np
import pandas as pd
'''统计重复出现的物种'''
def Re_Specise(path):
    Sp_value=[]
    with open(path) as f:
        all_list=f.read().split('\n')
    D={}
    ex_specise={}
    for item in range(len(all_list)-1):
        item_list=all_list[item].split(',')
        Sp_value.append(item_list[1])
        ex_temp=[]
        for i in range(2,11):
            item_list[i]=item_list[i].replace("'", '')
            item_list[i]=item_list[i].replace('[','')
            item_list[i] = item_list[i].replace(']', '')
            ex_temp.append(item_list[i])
            if item_list[i] not in D.keys():
                D[item_list[i]]=1
            else:
                D[item_list[i]]=D[item_list[i]]+1
        ex_specise[item+1]=ex_temp
        df=pd.DataFrame(ex_specise).T
    print(D)
    return Sp_value,df

def main():
    path='C:/Users/97899/Desktop/Biomass/Spearman.txt'
    Sp,ex_specise=Re_Specise(path)
    print(ex_specise)
main()