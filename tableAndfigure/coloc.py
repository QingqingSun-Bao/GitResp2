import matplotlib.pyplot as plt
import numpy as np

# x=np.linspace(0,1,10)
# fig,ax=plt.subplots()
# ax.set_color_cycle(["red","black","yellow"])
# for i in range(1,6):
#     plt.plot(x,i*x+i)
# plt.legend()
# plt.show()

category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85))
print(category_colors)