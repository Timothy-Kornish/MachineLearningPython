import pandas as pd

ecom = pd.read_csv('EcommercePurchases.csv')
print(ecom.head())

rowsNum = len(ecom['Address'])
columnsNum = len(ecom.columns)
print('+---================---+')
print("data frame info: \n", ecom.info())
print("number of rows: ", rowsNum)
print("number of columns: ", columnsNum)
print('+---================---+')

avePurchasePrice = ecom['Purchase Price'].mean()
print("average purchase price: ", avePurchasePrice)
print("highest purchase price: ", ecom['Purchase Price'].max())
print("lowest purchase price: ", ecom['Purchase Price'].min())

def search_count(column, keyWord):
    if keyWord.lower() in column.lower():
        return True
    else :
        return False

englishCount = sum(ecom['Language'].apply(lambda x: search_count(x, 'en')))
print("english speaker total: ", englishCount)
lawyers = sum(ecom['Job'].apply(lambda x: search_count(x, 'Lawyer')))
print("Number of lawyers: ", lawyers)
print('+---================---+')
print("purchases in AM vs. PM: \n", ecom['AM or PM'].value_counts())
print('Top 5 most commond jobs:\n', ecom['Job'].value_counts()[:5])
print('+---================---+')
purchaseFromLot90WT = ecom[ecom['Lot'] == '90 WT']['Purchase Price']
print("purhcase from lot 90 Wt: ", purchaseFromLot90WT)

email = ecom[ecom['Credit Card'] == 4926535242672853]['Email']
print("special email: ", email)
AE95AndUp = ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count()
print(AE95AndUp)

CCExpires2025 = sum(ecom['CC Exp Date'].apply(lambda x: search_count(x, '25')))
print('+---================---+')
print("number of cards that expire in 2025: ", CCExpires2025)

CommonEmailProvider = ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts()[:5]

print("Top 5 email providers:\n", CommonEmailProvider)
