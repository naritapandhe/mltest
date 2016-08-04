import pandas
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


#X = [[0., 0.], [1., 1.]]
#y = [0, 1]


df=pandas.read_json("updatedFinalData.json")


X = (df[["identifier"]])
X_data = np.array(X["identifier"])
#print X_train
#print X_train.shape

Y = (df[["labels"]])
Y_data = Y["labels"]
#print Y_data
#print Y.shape


#for x in X_train:
#	print x
lb = preprocessing.MultiLabelBinarizer()
Y_data_transformed = lb.fit_transform(Y_data)
#print Y_data_transformed


"""skf = KFold(len(X_data), n_folds=10,shuffle=True,random_state=42)
for train_index, test_index in skf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data_transformed[train_index], Y_data_transformed[test_index]"""

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_data, Y_data_transformed, test_size=0.3, random_state=None)



classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', (MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(35, 32), random_state=42)))])
    #('clf', OneVsRestClassifier(LinearSVC(C=1.)))])


"""
clf = 
clf.fit(X, y) 
print(clf.predict([[2., 2.], [-1., -2.]]))
"""

classifier.fit(X_train, y_train)

joblib.dump(classifier,'mlpmodel.pkl')

