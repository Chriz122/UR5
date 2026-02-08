import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plotCalbPoint3d_noText(xy):
    num = len(xy)
    y = xy[:, 0]
    x = -xy[:, 1]
    z = xy[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='+')
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)
    ax.set_xlim([xmin - 5, xmax + 5])
    ax.set_ylim([ymin - 5, ymax + 5])
    ax.set_zlim([zmin - 5, zmax + 5])
    ax.grid(True)
    # 無文字標籤
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    plt.show()