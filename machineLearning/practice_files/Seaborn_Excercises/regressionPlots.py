import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

tips = sns.load_dataset('tips')

#-------------------------------------------------------------------------------
#                         LM Plots
"""
hue: makes two plots overlap
col: makes seperate plots, one for each hue
row: makes seperate plots, one for each hue
"""
#-------------------------------------------------------------------------------

lm = sns.lmplot(x = 'total_bill', y = 'tip', data = tips, hue = 'sex',
                markers = ['o', 'v'], scatter_kws = {'s':20})
plt.show(lm)

#
duoLM= sns.lmplot(x = 'total_bill', y = 'tip', data = tips,
                  col = 'day', row = 'time', hue = 'sex', size = 2)
plt.show(duoLM)
