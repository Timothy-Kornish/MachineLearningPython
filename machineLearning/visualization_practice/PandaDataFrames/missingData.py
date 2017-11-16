import numpy as np
import pandas as pd

d = {'A' : [1, 2, np.nan], 'B' : [5, np.nan, np.nan], 'C' : [1, 2, 3]}
df = pd.DataFrame(d)

#dropping missing values

print('---------------\nData Frame with NaN\'s:')
print(df)

"""

print('---------------\nrows without NaN:')
print(df.dropna(axis = 0)) # inplace = True permenantly deletes
print('---------------\ncolumns without NaN:')
print(df.dropna(axis = 1))

print('---------------\nrows with threshold of 2 real numbers minimum:')
print(df.dropna(thresh = 2))
print('---------------\nrows with threshold of 1 real number minimum:')
print(df.dropna(thresh = 1))

print('---------------\nColumns with threshold of 2 real numbers minimum:')
print(df.dropna(thresh = 2, axis = 1))
print('---------------\nColumns with threshold of 1 real number minimum:')
print(df.dropna(thresh = 1, axis = 1))

"""

# filling in missing arguments
print('-------------\nFilling in NaN with values:')
print(df.fillna(value = 'FILL VALUE'))
print('-------------\nFilling in NaN  in column A with mean of column:')
print(df['A'].fillna(value = df['A'].mean()))
