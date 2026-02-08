import matplotlib.pyplot as plt
import numpy as np

def plotCalbPointErr(err, rn, cn):
    N = rn * cn
    num = len(err)
    x_mesh, y_mesh = np.meshgrid(np.arange(1, cn + 1), np.arange(-rn, 0))
    x = x_mesh.reshape(1, -1).flatten()
    y = y_mesh.reshape(1, -1).flatten()
    m = 1
    ct = 1
    cEr = ['-xErr', '-yErr', '-zErr']
    fig_count = 1
    for n in range(0, num, N):
        sct = str(ct)
        for q in range(3):
            plt.subplot(3, 3, (m-1)*3 + q + 1)
            plt.plot(x, y, '+')
            plt.title(sct + cEr[q])
            for p in range(N):
                if n + p < num:
                    Er = err[n + p, q]
                    sEr = str(round(Er, 3))
                    if Er <= 1:
                        color = 'black'
                        fontsize = 4
                    elif Er <= 10:
                        color = 'blue'
                        fontsize = 6
                    elif Er <= 20:
                        color = 'red'
                        fontsize = 6
                    else:
                        color = 'magenta'
                        fontsize = 6
                    plt.text(x[p], y[p], sEr, fontsize=fontsize, color=color)
        if m % 3 == 0:
            plt.figure(fig_count)
            fig_count += 1
            m = 0
        m += 1
        ct += 1
    plt.show()