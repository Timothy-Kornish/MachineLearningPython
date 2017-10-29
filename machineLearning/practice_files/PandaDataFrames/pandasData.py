import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'] , ['W', 'X', 'Y', 'Z'])
"""
print(df)
print('---------')
print(df['W'])
print(type(df))
print('---------')
print(df.W) # avoid as built in methods may match column names
print('---------')
print(df[['W', 'Z']]) # for columns in data frame
print('---------')
"""
df['new'] = df['W'] + df['Y']
"""
print(df)
print(df.shape)
df.drop('new', axis = 1, inplace = True) # axis = 1 for columns
print('------------------------------------')
print(df)
df.drop('E', axis = 0, inplace = True) # axis = 0 for rows
print('------------------------------------')
print(df)
print(df.shape)
"""

print('row E:\n', df.loc['E']) # for rows in data frame
print("index 2:\n", df.iloc[2])
print('value at b,y:\n', df.loc['B', 'Y'])
print(df.loc[['A', 'B'], ['W', 'Y']])
