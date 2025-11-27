import matplotlib.pyplot as plt
import numpy as np

def plotCalbPoint(xy):
    num = len(xy)
    x = xy[:, 0]
    y = -xy[:, 1]
    plt.plot(x, y, '+')
    plt.axis([0, 640, -480, 0])

    for n in range(num):
        sn = str(n + 1)  # MATLAB 是 1 索引
        plt.text(x[n], y[n], sn, fontsize=6)

    plt.show()