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
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


#nltk.download_shell() #download stopwords, only needs to be run once

#-------------------------------------------------------------------------------
#           Natural Language Processing for an email spam filter
#-------------------------------------------------------------------------------

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print("\n-------------------------------------------------------------------\n")
print("number of messages in spamcollection: ",len(messages))
print("\n-------------------------------------------------------------------\n")

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message, '\n')
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#       Make Data Frame of tab seperated value file, (tsv instead of csv)
#-------------------------------------------------------------------------------

messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep = '\t', names = ['label', 'message'])
print(messages.head())
print("\n-------------------------------------------------------------------\n")
print(messages.describe())
print("\n-------------------------------------------------------------------\n")
print(messages.groupby('label').describe())
print("\n-------------------------------------------------------------------\n")

messages['length'] = messages['message'].apply(len)
print(messages.head())
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#   plotting histogram of message length, showing bimodal behavior (two peaks)
#-------------------------------------------------------------------------------

messages['length'].hist(bins = 100)
plt.xlabel("length of message")
plt.show()

#-------------------------------------------------------------------------------
#       plotting pandas version of seaborns facet grid with histograms
#-------------------------------------------------------------------------------

messages.hist(column = 'length', by = 'label', bins = 60, figsize = (12, 4))
plt.xlabel("length of message")
plt.show()

#-------------------------------------------------------------------------------
#                printing the longest message sent
#-------------------------------------------------------------------------------

print(messages[messages['length'] == messages['length'].max()])
print("\n-------------------------------------------------------------------\n")
print("full message:\n", messages[messages['length'] == messages['length'].max()]['message'].iloc[0] )
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#                        Text Pre Processing
#-------------------------------------------------------------------------------

mess = 'Simple message! Notice: it has punctuation.'
print("string puncutaions: ", string.punctuation)
print("\n-------------------------------------------------------------------\n")

# list comprehension, c will be a new index for every iteration in for loop
no_punc = [c for c in mess if c not in string.punctuation]
print("mess with punctuations removed: ", no_punc)
print("\n-------------------------------------------------------------------\n")

no_punc = ''.join(no_punc)
print("mess with punctuations removed, in string form: ", no_punc)
print("\n-------------------------------------------------------------------\n")

unhelpful_words = stopwords.words('english')
print("words that don't help distinguish spam from ham as they are too common:\n", unhelpful_words )
print("\n-------------------------------------------------------------------\n")

clean_mess = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
print("no_punc array with removed stopwords: ", clean_mess )
print("\n-------------------------------------------------------------------\n")

def text_process(mess):
    """
    1. remove punctuations
    2. remove stop words
    3. return list of clean text words
    """
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower not in stopwords.words('english')]

#-------------------------------------------------------------------------------
#                Tokenizing text strings in message data frame
#-------------------------------------------------------------------------------

print("tokenized messages:\n", messages['message'].head(5).apply(text_process))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#  Converting tokens into vectors that machine learning models can understand
#  Using a sparse matrix as most values will be 0
# bow = bag of words
#-------------------------------------------------------------------------------

bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
print("\n-------------------------------------------------------------------\n")

mess4 = messages['message'][3]
bow4 = bow_transformer.transform([mess4])

print("4th message: ", mess4)
print("\n-------------------------------------------------------------------\n")
print(bow4)
print("\n-------------------------------------------------------------------\n")
print("shape of bow4:\n", bow4.shape)
print('Showing repeated words:\n', bow_transformer.get_feature_names()[4068],
                              " ", bow_transformer.get_feature_names()[9832])

#-------------------------------------------------------------------------------
#  Creating sparse matrix of messages
#-------------------------------------------------------------------------------

messages_bow = bow_transformer.transform(messages['message'])
print("\n-------------------------------------------------------------------\n")
print("Shape of sparse matrix: ", messages_bow.shape)
print("\n-------------------------------------------------------------------\n")
print("Number of non-zero occurences in sparse matrix: ", messages_bow.nnz)
print("\n-------------------------------------------------------------------\n")

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#  TF-IDF stands for term frequency-inverse document frequency
#-------------------------------------------------------------------------------

tf_idf_transformer = TfidfTransformer().fit(messages_bow)

tf_idf4 = tf_idf_transformer.transform(bow4)
print("Inverse document frequency:\n", tf_idf4)
print("\n-------------------------------------------------------------------\n")

print("Inverse document frequency on word \"university\": ",
      tf_idf_transformer.idf_[bow_transformer.vocabulary_['university']])
print("\n-------------------------------------------------------------------\n")

messages_tf_idf = tf_idf_transformer.transform(messages_bow)

#-------------------------------------------------------------------------------
#  Multinomial Naive Bayes Classifier
#-------------------------------------------------------------------------------

spam_detect_model = MultinomialNB().fit(messages_tf_idf, messages['label'])
all_predictions = spam_detect_model.predict(messages_tf_idf) # done on entire data set, not graph_from_dot_data

#-------------------------------------------------------------------------------
#  training and testing data
#-------------------------------------------------------------------------------

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.3)

#-------------------------------------------------------------------------------
#  Data pipeline from sklearn for MultinomialNB
# creates pipeline to perform multiple steps automatically one after the other
#-------------------------------------------------------------------------------

pipeline = Pipeline(steps = [
                    ('bow', CountVectorizer(analyzer = text_process)),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())
                    ])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

print("Classification report on message pipeline training data with Multinomial Naive Bayes:\n",
       classification_report(label_test, predictions))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#  Data pipeline from sklearn for Random Forest
#-------------------------------------------------------------------------------

pipeline_forest = Pipeline(steps = [
                    ('bow', CountVectorizer(analyzer = text_process)),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', RandomForestClassifier())
                    ])

pipeline_forest.fit(msg_train, label_train)
predictions_forest = pipeline_forest.predict(msg_test)

print("Classification report on message pipeline training data with Random Forest:\n",
       classification_report(label_test, predictions_forest))
print("\n-------------------------------------------------------------------\n")
