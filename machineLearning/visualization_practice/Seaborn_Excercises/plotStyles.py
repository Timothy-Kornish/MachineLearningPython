import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#-------------------------------------------------------------------------------
#                        Plot styles
"""
set_style():
        white:      blank basic background with plot in a basic black square
        ticks:      plot in a black square with ticks at each axis value
        darkgrid:   No black square surrounding plot, now has grey rows seperating y axis values
        whitegrid:  No black square surrounding plot, now has white rows seperating y axis values

despine():
        default: removes right and top borders (spines)
        bool params: top, bottom, left, right, trim
        numeric params: offset, raises bars above bottom line

plt.figure(figsize = (12, 3)):
        changes dimensions of plot to wide and short

set_context():
        poster: makes font larger to be printed on poster
        notebook: default display
        param:
            font_scale: change size of font

"""
#-------------------------------------------------------------------------------

tips = sns.load_dataset('tips')

sns.set_style('ticks')
count = sns.countplot(x = 'sex', data = tips)
#sns.despine(offset = 4)
#sns.set_context('poster', font_scale = 2)
plt.show(count)

lm = sns.lmplot(x = 'total_bill', y = 'tip', data = tips, hue = 'sex', palette = 'seismic')
plt.show(lm)
