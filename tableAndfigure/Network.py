import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

'''导入数据'''


def LoadDataSet(path):
    all_array = np.loadtxt(path)
    C_mat = np.mat(all_array[0:9])
    return C_mat







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

def get_edge(C_mat,node):
    edge=[]
    M=C_mat.shape[0]
    for i in range(M):
        for j in range(M):
            if C_mat[i,j]>0.8:
                edge.append((node[i],node[j]))
    return edge



def main():
    path = 'C:/Users/97899/Desktop/N/Biomass/Spearman.txt'
    Sp, ex_specise = Re_Specise(path)
    print(ex_specise)
    for i in range(1, 39):
        print('第'+str(i)+'个实验')
        ex = float(i)
        path = 'C:/Users/97899/Desktop/Biomass/' + str(ex) + '.txt'
        C_mat = LoadDataSet(path)
        node_list=ex_specise.iloc[i - 1, :]
        # print(pd.DataFrame(C_mat))
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']
        G = nx.DiGraph()
        G.add_nodes_from(node_list)  # 添加点a
        edge_list=get_edge(C_mat,node_list)
        G.add_edges_from(edge_list)#添加边,起点为x，终点为y
        '''显示图形'''
        nx.draw(G,pos=nx.circular_layout(G),node_color = 'lightgreen',edge_color = 'black',with_labels = True,font_size =10,node_size =3000)
        # plt.title('第'+str(ex)+'个实验')
        plt.show()

main()