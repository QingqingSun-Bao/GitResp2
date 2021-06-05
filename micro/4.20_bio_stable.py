# -*- coding: utf-8 -*-
# @Time:2021/4/2010:03
# @File:4.20_bio_stable.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    path="C:/Users/97899/Desktop/N/"
    df_bio=pd.read_excel(path+"Biomass/bio_ex21.xls")
    df_bio.set_index("Unnamed: 0",inplace=True)
    df_comp=pd.read_excel(path+"Network/Strong_index.xls")
    df_comp.set_index("Unnamed: 0",inplace=True)
    df_plot=dict((k,[]) for k in ["loop","chain","ed"])
    ind =np.linspace(2008,2020,13)
    for year in ind:
        loop_bio=[]
        chain_bio=[]
        ed_bio=[]
        for ex in range(1,39):
            if df_comp.loc[ex-1,year]>0:
                loop_bio.append(df_bio.loc[ex-1,year])
            if df_comp.loc[ex-1,year]==0:
                chain_bio.append(df_bio.loc[ex-1, year])
            if df_comp.loc[ex-1,year]==-0.15:
                ed_bio.append(df_bio.loc[ex - 1, year])

        df_plot["loop"].append(np.mean(loop_bio))
        df_plot["chain"].append(np.mean(chain_bio))
        df_plot["ed"].append(np.mean(ed_bio))

    plt.plot(ind,df_plot["loop"],label="loop")
    plt.plot(ind,df_plot["chain"],label="chain")
    # plt.plot(ind,df_plot["ed"],label="ed")
    plt.legend()
    plt.show()