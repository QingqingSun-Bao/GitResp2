# -*- coding: utf-8 -*-
# @Time:2021/3/299:08
# @File:Fig14.weather_year.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rain_df=pd.read_excel("C:/Users/97899/Desktop/N/Enveriment/weather_temp_20.xls")
x=np.linspace(2008,2020,13)
plt.plot(x,rain_df.iloc[:,3],"turquoise",label="Annual temperature") #precipitation
plt.plot(x,rain_df.iloc[:,1],"darkcyan",label="Growing season temperature")
# plt.xlim([x])
plt.xticks(x)
plt.ylabel("")
plt.legend()
plt.show()