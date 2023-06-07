"""
Created on junho 06 20:29:26 2023

@author: Ânderson Felipe Weschenfelder
"""
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


df_todos = X
df_todos['isMato'] = y

#----MEDIAS----

#MEDIA 0
semMato = np.array(df_todos[['media0']].iloc[list(y == 0)]).flatten()
comMato = np.array(df_todos[['media0']].iloc[list(y == 1)]).flatten()

plt.figure()
bp = plt.boxplot([semMato, comMato], positions = [1, 2], widths = 0.6, showfliers=False)
setBoxColors(bp)

#MEDIA 1
semMato = np.array(df_todos[['media1']].iloc[list(y == 0)]).flatten()
comMato = np.array(df_todos[['media1']].iloc[list(y == 1)]).flatten()

bp = plt.boxplot([semMato, comMato],  positions = [4, 5], widths = 0.6, showfliers=False)
setBoxColors(bp)

#MEDIA 2
semMato = np.array(df_todos[['media2']].iloc[list(y == 0)]).flatten()
comMato = np.array(df_todos[['media2']].iloc[list(y == 1)]).flatten()

bp = plt.boxplot([semMato, comMato],  positions = [7, 8], widths = 0.6, showfliers=False)
setBoxColors(bp)

#MEDIA 3
semMato = np.array(df_todos[['media3']].iloc[list(y == 0)]).flatten()
comMato = np.array(df_todos[['media3']].iloc[list(y == 1)]).flatten()

bp = plt.boxplot([semMato, comMato],  positions = [10, 11], widths = 0.6, showfliers=False)
setBoxColors(bp)

#MEDIA 4
semMato = np.array(df_todos[['media4']].iloc[list(y == 0)]).flatten()
comMato = np.array(df_todos[['media4']].iloc[list(y == 1)]).flatten()

bp = plt.boxplot([semMato, comMato],  positions = [13, 14], widths = 0.6, showfliers=False)
setBoxColors(bp)

#MEDIA 5
semMato = np.array(df_todos[['media5']].iloc[list(y == 0)]).flatten()
comMato = np.array(df_todos[['media5']].iloc[list(y == 1)]).flatten()

bp = plt.boxplot([semMato, comMato],  positions = [16, 17], widths = 0.6, showfliers=False)
setBoxColors(bp)

plt.xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5],['Mean 0', 'Mean 1', 'Mean 2', 'Mean 3', 'Mean 4', 'Mean 5'])

hB, = plt.plot([1,1],'b-')
hR, = plt.plot([1,1],'r-')
plt.legend((hB, hR),('Without', 'With'), loc = 'upper right')
hB.set_visible(False)
hR.set_visible(False)