import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

k=30#此处为大小，由于3/5/7数据过小，无法准确绘制图像，所以放大10倍
x,y = np.mgrid[-1:1:2/k,-1:1:2/k]
sigma=1#此处为方差
z = 1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2))


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
print(z)