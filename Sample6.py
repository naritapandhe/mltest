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

df=pandas.read_json("updatedFinalData.json")


X = (df[["identifier"]])
X_data = np.array(X["identifier"])
#print X_train
#print X_train.shape

Y = np.array(df[["gadget_id"]])
print Y
print Y.shape

#for x in X_train:
#	print x
lb = preprocessing.MultiLabelBinarizer()
Y_data_transformed = lb.fit_transform(Y)
print Y_data_transformed

kf = KFold(len(X_data), n_folds=10)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data_transformed[train_index], Y_data_transformed[test_index]

#X_train, X_test, y_train, y_test = sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,random_state=None)
#train_test_split(X_data, Y_data_transformed, test_size=0.2, random_state=5)



classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(C=10.)))])

classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)

#print all_labels
for item, labels in zip(X_test, all_labels):
    print (item)
    print labels


print (classifier.score(X_train,y_train))  
print (classifier.score(X_test,y_test))    

