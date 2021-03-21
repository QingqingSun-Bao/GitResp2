import pandas as pd
import numpy as np
import seaborn as sb
import statsmodels.api as sm
import matplotlib.pyplot as plt

path = "C:/Users/97899/Desktop/N/"
df_Int = pd.read_excel("C:/Users/97899/Desktop/U/Intransitivity/Tao_C.xls", sheet_name="Int")
df_biomass = pd.read_excel(path + "Biomass/bio_ex.xls")
ind = np.linspace(2009, 2019, 11).tolist()
Int = []
bio = []
for jtem in ind:
    for item in range(38):
        if df_Int.loc[item, jtem] > -0.15:
            Int.append(df_Int.loc[item, jtem])
            bio.append(np.log(df_biomass.loc[item, jtem]))
            if df_biomass.loc[item, jtem] is np.nan:
                print(item, jtem)
print(Int, bio)
x = sm.add_constant(bio)
model = sm.OLS(Int, x)
results = model.fit()
print(results.summary())
y_fitted = results.fittedvalues
plt.scatter(bio, Int, c='b', s=50, marker='o')
sb.regplot(bio, Int, fit_reg=True, x_jitter=0.3, y_jitter=0.05, scatter_kws={"alpha": 1 / 3})
plt.text(3000, 0.9, r"$r^2=0.009$", fontsize=13)
plt.ylabel('Intransitivity', fontsize=13)
plt.xlabel("Biomass", fontsize=13)
# plt.plot(x, y_fitted, 'r')
# pd.DataFrame((x,Int)).T.to_excel('C:/Users/97899/Desktop/N/Int_Rich.xls')
plt.ylim(0, 1.1)
plt.show()
