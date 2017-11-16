import numpy as np
import pandas as pd
from numpy.random import randn
# randn vs rand
# randn is the standard normal distribution centered at 0 from -inf to inf.
# rand is between 0 and 1

np.random.seed(101)

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'] , ['W', 'X', 'Y', 'Z'])

resultdf = df[df['W'] > 0]
#conditional selection with bracket notation
# one condition
"""
print(df)
print('------------------')
booldf = df > 0
print(booldf)
print('------------------')
print(df[booldf])
print('------------------')
print(df['W'] > 0)
print('------------------')

print(resultdf) # no more null values NaN
# asking for rows where W > 0 is true
print('------------------')
print(df[df['Z'] < 0]) # no more null values NaN
# asking for rows where Z < 0 is true
print('------------------')
print(resultdf['X'])
print('------------------')
print(df[df['W'] > 0][['X', 'W']])
"""

boolser = df['W'] > 0
result = df[boolser]
my_cols = ['X', 'Y']
print(boolser)
print('------------------')
print(result[my_cols])
print('------------------')
#multiple conditions in bracket notation

print(df[(df['W'] >0) & (df['Y'] > 0)]) # only single &, no 'and' for list of booleans
# 'and' is meant for single boolean comparison, not lists
print('------------------')
print(df[(df['W'] >0) | (df['Y'] > 0)]) # only single |, no 'or' for list of booleans
# 'or' is meant for single boolean comparison, not lists
print('------------------')
print(df.reset_index())
print('------------------')
print(df)
print('------------------')
newInd = 'CA NY WY OR CO'.split()
print(newInd)
df['States'] = newInd
print('------------------')
print(df)
print('------------------')
print(df.set_index('States')) # overwrites index names column
