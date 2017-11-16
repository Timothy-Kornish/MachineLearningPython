import numpy as np
import pandas as pd
labels = ['a', 'b', 'c']

my_data = [10, 20, 30]

arr = np.array(my_data)

dic = {'a' : 10, 'b' : 20, 'c' : 30}


"""
print(pd.Series(data = my_data))
print('----------------------')
print(pd.Series(data = my_data, index = labels))
print('----------------------')
print(pd.Series(my_data, labels))
print('----------------------')
print(pd.Series(arr))
print('----------------------')
print(pd.Series(dic))
print('----------------------')
print(pd.Series([sum, print, len]))
"""

ser1 = pd.Series([1,2,3,4], ['USA', 'Germany', 'USSR', 'Japan'])
ser2 = pd.Series([1,2,5,4], ['USA', 'Germany', 'Italy', 'Japan'])
print(ser1)
print(ser2)
print("USA  = ", ser1['USA'])

ser3 = pd.Series(data = labels)
print(ser3[0])

print(ser1 + ser2)
