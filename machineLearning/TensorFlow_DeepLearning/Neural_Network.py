import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


iris = load_iris()
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(hidden_units = [10, 20, 10], n_classes = 3, feature_columns = feature_columns)

X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

classifier.fit(X_train, y_train, steps = 200, batch_size = 32)
iris_predictions = classifier.predict(X_test, as_iterable = False)
print("\n-------------------------------------------------------------------\n")
print("Predictions: ", iris_predictions)
print("\n-------------------------------------------------------------------\n")
print("Classification report for Deep Neural Network:\n", classification_report(y_test, iris_predictions))
print("\n-------------------------------------------------------------------\n")
print("Confusion matrix for Deep Neural Network:\n", confusion_matrix(y_test, iris_predictions))
