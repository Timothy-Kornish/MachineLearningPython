import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#plotly iplot can only be used in jupyter notebook unfortunately

init_notebook_mode(connected = True)
#plotly connects pandas and python to javascript library

cf.go_offline()
#makes cufflinks work offline

# Data

df1 = pd.DataFrame(np.random.randn(100, 4), columns = 'A B C D'.split())
df2 = pd.DataFrame({"Category": ['A', 'B', 'C'], "values": [32, 43, 50]})

#Shows line plot of dataframe
df1.plot()
plt.show()

# iplot makes plot interactive, i.e. iplot = interactive plot
df1.iplot()
plt.show()

#rest of this will be done on jupyter notebook
