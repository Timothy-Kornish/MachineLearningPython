import pandas as pd
import numpy as np

#------------------------------------------------------------------------------
#                             Concatenation
#------------------------------------------------------------------------------

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

"""

print('====================\nSingle Data Frames:\n')
print(df1, '\n---------------\n', df2, '\n-----------------\n', df3)
print('====================\nConcatenated Data Frames by rows:\n')
print(pd.concat([df1, df2, df3]))
print('====================\nConcatenated Data Frames by columns with missing indexes:\n')
print(pd.concat([df1, df2, df3], axis = 1))

"""

#------------------------------------------------------------------------------
#                             Merging
#------------------------------------------------------------------------------

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})

"""

print('====================\nSingle Data Frames:\n')
print(left, '\n---------------\n', right)
print('====================\nSingle Inner Merged Data Frame:\n')
print(pd.merge(left, right, how = 'inner', on = 'key'))

"""

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})

"""

print('====================\nSingle Data Frames:\n')
print(left, '\n---------------\n', right)
print('====================\nSingle Merged Data Frame with 2 keys:\n')
print(pd.merge(left, right, on = ['key1', 'key2']))
print('====================\nSingle Outer Merged Data Frame :\n')
print(pd.merge(left, right, how = 'outer', on = ['key1', 'key2']))
print('====================\nSingle Right Merged Data Frame :\n')
print(pd.merge(left, right, how = 'right', on = ['key1', 'key2']))
print('====================\nSingle Left Merged Data Frame :\n')
print(pd.merge(left, right, how = 'left', on = ['key1', 'key2']))

"""

#------------------------------------------------------------------------------
#                             Joining
#------------------------------------------------------------------------------

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])

print('====================\nSingle Data Frames:\n')
print(left, '\n---------------\n', right)
print('====================\nSingle left Joined right Data Frame:\n')
print(left.join(right))
print('====================\nSingle outer left Joined right Data Frame:\n')
print(left.join(right, how = 'outer'))
