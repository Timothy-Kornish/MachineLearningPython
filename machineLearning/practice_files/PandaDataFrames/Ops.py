import numpy as np
import pandas as pd

df = pd.DataFrame({'col1':[1,2,3,4],
                   'col2':[444,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()

#------------------------------------------------------------------------------
#                             Uniqueness
#------------------------------------------------------------------------------


"""
print('=================\n show only unique values of column expressed as array:')
print(df['col2'].unique(), '| type = ', type(df['col2'].unique()))
print('=================\n number of unique values:')
print(df['col2'].nunique())
print('=================\n number of occurences of each value:')
print(df['col2'].value_counts())
print('=================\n df: col1 > 2:')
print(df[(df['col1']>2)])
print('=================\n df: col1 > 2 and col2 == 444:')
print(df[(df['col1']>2) & (df['col2'] == 444)])
"""

#------------------------------------------------------------------------------
#                applying normal and lambda fuctions to df
#------------------------------------------------------------------------------

def times2(x):
    return x * 2

"""

print('=================\n values of col1 times 2:')
print(df['col1'].apply(times2))
print('=================\n len of values in col3 :')
print(df['col3'].apply(len))
print('=================\n lambda expression, value in col2 * 3 :')
print(df['col2'].apply(lambda x: x * 3))
print('=================\n show columns:')
#df.drop('col1', axis = 1, inplace = True)
print(df.columns)
print('=================\n show index:')
print(df.index)

"""

#------------------------------------------------------------------------------
#                             sorting
#------------------------------------------------------------------------------

print('=================\n table sorted by col2:')
print(df.sort_values(by = 'col2'))
print('=================\n boolean table with nulls:')
print(df.isnull())


data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
print('=================\n New table:')
print(df)
print('=================\n sorted table via column C:')
print(df.pivot_table(values = 'D', index = ['A', 'B'], columns = ['C']))
