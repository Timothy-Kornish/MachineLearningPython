import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#-------------------------------------------------------------------------------
#           Natural Language Processing for classifying Yelp reviews
#-------------------------------------------------------------------------------

yelp = pd.read_csv('yelp.csv')

print("\n-------------------------------------------------------------------\n")
print(yelp.head())
print("\n-------------------------------------------------------------------\n")
print(yelp.describe())
print("\n-------------------------------------------------------------------\n")
print(yelp.info())
print("\n-------------------------------------------------------------------\n")

yelp['text length'] = yelp['text'].apply(len)

print(yelp.head())
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           PLotting histograms of text length based on star rating
#-------------------------------------------------------------------------------

"""
this will stack them in a matrix like format that looks ugly, use seaborn instead

yelp.hist(column = 'text length', by = 'stars', bins = 50, figsize = (14, 6))
plt.show()
"""

facet = sns.FacetGrid(data = yelp, col = 'stars')
facet.map(plt.hist, 'text length')
plt.show(facet)

#-------------------------------------------------------------------------------
#           Plotting box-plots of text length based on star rating
#-------------------------------------------------------------------------------

sns.boxplot(data = yelp, x = 'stars', y = 'text length', palette = 'rainbow')
plt.show()

#-------------------------------------------------------------------------------
#           Plotting count plot for each star rating
#-------------------------------------------------------------------------------

sns.countplot(x = 'stars', data = yelp)
plt.show()

stars_mean = yelp.groupby('stars').mean()
print("data frame of means for each star's rating:\n", stars_mean)
print("\n-------------------------------------------------------------------\n")

print(" correlation data frame of means for each star's rating:\n", stars_mean.corr())
print("\n-------------------------------------------------------------------\n")


#-------------------------------------------------------------------------------
#           heat map of corr() dataframe
#-------------------------------------------------------------------------------

sns.heatmap(stars_mean.corr(), cmap = 'coolwarm', annot = True)
plt.show()

#-------------------------------------------------------------------------------
#           NLP Classification - Train Test Split
#-------------------------------------------------------------------------------

yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

X = yelp_class['text']
y = yelp_class['stars']

countVec = CountVectorizer()
X = countVec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

nb = MultinomialNB()
nb.fit(X_train, y_train)

pred = nb.predict(X_test)

print("Classification Report with Multinomial naive bayes:\n",
       classification_report(y_test, pred))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix with Multinomial naive bayes:\n",
       confusion_matrix(y_test, pred))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#        pipeline to automate several steps using MultinomialNB
#-------------------------------------------------------------------------------

pipeline = Pipeline(steps = [
                            ('bow', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('classifier', MultinomialNB())
                            ])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)
pipe_pred = pipeline.predict(X_test)

print("Classification Report through a pipeline with Multinomial naive bayes:\n",
       classification_report(y_test, pipe_pred))
print("Notice that the Tf-Idf made things worse for this data")
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix through a pipeline with Multinomial naive bayes:\n",
       confusion_matrix(y_test, pipe_pred))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#     pipeline to automate several steps using a random forest
#-------------------------------------------------------------------------------

pipeline = Pipeline(steps = [
                            ('bow', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('classifier', RandomForestClassifier())
                            ])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)
pipe_pred = pipeline.predict(X_test)

print("Classification Report through a pipeline with Random Forest:\n",
       classification_report(y_test, pipe_pred))
print("Notice that the Random Forest made things better for this data")
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix through a pipeline with Random Forest:\n",
       confusion_matrix(y_test, pipe_pred))
print("\n-------------------------------------------------------------------\n")
