import numpy as np
import pandas as pd

# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)
print('---------------------------\nGroup By data methods')
print(df)
print('---------------------------\nGroup method, grab mean sales of each Company, ignore NaN\'s')
byComp = df.groupby('Company')
print(byComp.mean())
print('---------------------------\nGroup method, grab sum of each Company sales, ignore NaN\'s')
print(byComp.mean())
print('---------------------------\nGroup method, grab std deviation of each Company sales, ignore NaN\'s')
print(byComp.std())
print('---------------------------\nGroup method, sum of sales for specific company')
print(byComp.sum().loc['FB'])
print('---------------------------\n same method, single line form, its more memory conservative')
print(df.groupby("Company").sum().loc['FB'])

print('---------------------------\n GroupBy method, count instances in each group')
print(df.groupby('Company').count())
print('---------------------------\n GroupBy method, count max in each group')
print(df.groupby('Company').max())
print('---------------------------\n GroupBy method, count min in each group')
print(df.groupby('Company').min())

print('---------------------------\n GroupBy method, describe table data')
print(df.groupby('Company').describe().transpose()['GOOG'])
