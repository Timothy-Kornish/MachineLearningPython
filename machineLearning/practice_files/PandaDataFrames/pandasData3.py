import numpy as np
import pandas as pd
from numpy.random import randn

np.random.seed(101)
#Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

print(outside)
print(inside)
print(hier_index)

df = pd.DataFrame(randn(6, 2), hier_index,['A','B'])
print('-----------------\n Data Frame:')
print(df)
print('-----------------\n G1:')
print(df.loc['G1'])
print('-----------------\n G1 row 1:')
print(df.loc['G1'].loc[1])
print('-----------------\n adding labels to Data Frame')
df.index.names = ["Groups", "Num"]
print(df)
print('-----------------\n G2 column B row 2 :')
print(df.loc['G2'].loc[2]['B'])
print('-----------------\n G2:')
print(df.xs(1, level = 'Num')) #xs is 'cross-section' data method
