import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('example.csv')

"""
print('+--==================--+\n Data Frame from csv:')
print(df)
print('+--==================--+\n Data Frame added column:')
print(df.to_csv('My_output', index = False))
print('+--==================--+\n Data Frame:')
print(pd.read_csv('example.csv'))
"""
# doesn't work because file is corrupted :(

#df = pd.read_excel('Excel_Sample.xlsx', sheetname = 'sheet1')
#print('+--==================--+\n Data Frame from Excel:')
#print(df)

df.to_excel('Example_Spreadsheet.xlsx', sheet_name = 'Sheet2')

data = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(data[0])

# data base for pandas
#PostgresSQL:  psycopg2
#MySQL:        pymysql
#SQLite: <Autoincluded in standard library>

engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)

sqlDf = pd.read_sql('my_table', con = engine)
print('+--==================--+\n SQLite Data Frame:')
print(sqlDf)
