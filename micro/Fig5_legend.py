# -*- coding: utf-8 -*-
# @Time:2021/3/2910:27
# @File:Fig5_legend.py
import matplotlib.pyplot as plt

color1N=["green","fuchsia"]
colorF=[ "blue","orange"]
colorM=["black","red"]
N1=["N<15","N>=15","M=No","M=Yes"]
# "F=Low","F=High"
# M=["M=No","M=Yes"]
# F=["F=0","F=Low","F=High"]
x=[1,2,3,4,5,6]
y=[6,3,4,2,5,8]
fig, ax1 = plt.subplots()
ax1.scatter(x,y, marker="o", color=color1N[0], facecolor="none" , s=50,label=N1[0])
ax1.scatter(x,y, marker="o", color=color1N[1], facecolor="none" , s=50,label=N1[1])
ax1.scatter(x,y, marker="o", color=colorM[0], facecolor="none" , s=50,label=N1[2])
ax1.scatter(x,y, marker="o", color=colorM[1], facecolor="none" , s=50,label=N1[3])
# ax1.scatter(x,y, marker="o", color=colorF[0], facecolor="none" , s=50,label=N1[4])
# ax1.scatter(x,y, marker="o", color=colorF[1], facecolor="none" , s=50,label=N1[5])
# ax1.scatter(x,y, marker="o", color=colorF[2], facecolor="none" , s=50,label=N1[6])
# plt.scatter(x,y,"o",facecolor="none",color=color1N[0],label=N1[0])
# plt.scatter(x,y,"o",facecolor="none",color=color1N[1],label=N1[1])
# plt.scatter(x,y,"o",facecolor="none",color=colorM[0],label=M[0])
# plt.scatter(x,y,"o",facecolor="none",color=colorM[1],label=M[1])
# plt.scatter(x,y,"o",facecolor="none",color=colorF[0],label=F[0])
# plt.scatter(x,y,"o",facecolor="none",color=colorF[1],label=F[1])
# plt.scatter(x,y,"o",facecolor="none",color=colorF[2],label=F[2])
ax1.legend(ncol=1, bbox_to_anchor=(1, 6), labels=N1)
plt.legend(ncol=1)
plt.show()




