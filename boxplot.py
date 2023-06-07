import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import setp

def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    #setp(bp['fliers'][0], color='blue')
    #setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    #setp(bp['fliers'][2], color='red')
    #setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

def plotboxplot(data):
        df_todos = data.copy()
        y = df_todos['Target']
        len = df_todos.shape
        a = 1
        for i in range(len[1] - 1):
                fire = np.array(df_todos[[i]].iloc[list(y == 0)]).flatten()
                nofire = np.array(df_todos[[i]].iloc[list(y == 1)]).flatten()
                # plt.figure()
                bp = plt.boxplot([fire, nofire], positions=[a, a + 1], widths=0.6, showfliers=False)
                a += 3
                setBoxColors(bp)

        plt.xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5, 31.5, 34.5, 37.5],
                   ['σ Red', 'σ Green','σ Blue', 'σ Global', r'$\mu$ Red', r'$\mu$ Green', r'$\mu$ Global', 'Median Global', 'Median Red', 'Mode Red', 'H Red', 'Min Red', 'entropy Red'], rotation='vertical')
        hB, = plt.plot([1, 1], 'b-')
        hR, = plt.plot([1, 1], 'r-')
        plt.legend((hB, hR), ('No-Fire', 'Fire'), loc='upper right')
        plt.title("Features")
        hB.set_visible(False)
        hR.set_visible(False)