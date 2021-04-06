# -*- coding: utf-8 -*-
# @Time:2021/4/113:13
# @File:Fig1_Loop_type.py
import networkx as nx
import matplotlib.pyplot as plt
ax = plt.gca()
G = nx.DiGraph()
node_list=["A","B","C","D","E","F","G","H"]
G.add_nodes_from(node_list)  # 添加点a
short_list=[("G","F"),("D","G"),("F","D")]
long_list=[("B","A"),("C","B"),("D","C"),("E","D"),("F","E"),("G","F"),("H","G"),("A","H")]
Inden_list=[("F","E"),("D","F"),("E","D"),("H","B"),("B","A"),("A","H")]
nest_list=[("D","F"),("B","A"),("C","B"),("D","C"),("E","D"),("F","E"),("G","F"),("H","G"),("A","H")]
cross_list=[("F","E"),("D","F"),("E","D"),("C","B"),("D","C"),("B","D")]
#edge_list = get_edge(C_mat, node_list)
G.add_edges_from(cross_list)  # 添加边,起点为x，终点为y
nx.draw(G, pos=nx.circular_layout(G,scale=0.2), node_color='lightblue',
        edge_color='red', with_labels=True,
        font_size=20, node_size=2000)
ax.set_title('Cross Loop',fontsize=20)
plt.show()
import matplotlib.pyplot as plt
import networkx as nx
#24，就是n=node，画椭圆
# G = nx.cycle_graph(24)#位置,
# iterations=迭代，建议200，
# 否则就不是椭圆pos = nx.spring_layout(G, iterations=200)
#
# #cmap=代表
# colormap=颜色图
# #nx.draw(G, pos, node_color=range(24), node_size=800, cmap=plt.cm.Blues)
# #nx.draw(G, pos, node_color=range(24), node_size=800, cmap=plt.cm.Reds)
# #with_labels=True 在节点内显示阿拉伯数字标签nx.draw(G, pos, node_color=range(24), node_size=800, cmap=plt.cm.Reds,with_labels=True)
# plt.show()