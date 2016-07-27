import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


"""X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york",
                    "mumbai is a hell of a town",
                    "mumbai is a humid",
                    "mumbai is in india",
                    "mumbai is in Maharashtra",
                    "mumbai is in like new york"])
y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
                ["new york"],["london"],["london"],["london"],["london"],
                ["london"],["london"],["new york","london"],["new york","london"],
                ["mumbai"],["mumbai"],["mumbai","india"],["mumbai"],
                ["mumbai","new york"]
                ]

X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too',
                   'hello welcome to mumbai. enjoy it here and mumbai too',
                   'mumbai is better than new york'
                   'Pune is India'])
target_names = ['New York', 'London','Mumbai','India']"""


X_train = np.array([
                    "iphone 6 att 16gb",
                    "iPhone 6 ATT 16GB",
                    "Apple iPhone 6 (ATT) 16GB",
                    "apple iphone 6 16gb att",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2889",
                    "iPhone 6 ATT 16 GB",
                    "iphone 6 verizon 16gb",
                    "iPhone 6 Verizon 16GB",
                    "apple iphone 6 verizon 16gb",
                    "Apple iPhone 6 (Verizon) 16GB",
                    "apple iphone 6 16gb verizon",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2898",
                    "iPhone 6 Verizon 16 GB",
                    "new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york",
                    "mumbai is a hell of a town",
                    "mumbai is a humid",
                    "mumbai is in india",
                    "mumbai is in Maharashtra",
                    "mumbai is in like new york",
                    "iphone 6 tmobile 16gb",
                    "iPhone 6 T Mobile 16GB",
                    "Apple iPhone 6 (T Mobile) 16GB",
                    "apple iphone 6 16gb t mobile",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2895",
                    "iPhone 6 T Mobile 16 GB",
                    "Apple 6 16gb T Mobile",
                    "iphone 6 sprint 16gb",
                    "iPhone 6 Sprint 16GB",
                    "apple iphone 6 sprint 16gb",
                    "Apple iPhone 6 (Sprint) 16GB",
                    "apple iphone 6 16gb sprint",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2892",
                    "iphone 6 unlocked 16gb",
                    "iPhone 6 Unlocked 16GB",
                    "Apple iPhone 6 (Factory Unlocked) 16GB",
                    "apple iphone 6 16gb unlocked",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2904",
                    "iPhone 6 Plus Factory Unlocked 16 GB",
                    "iphone 6 att 128gb",
                    "iPhone 6 ATT 128GB",
                    "apple iphone 6 att 128gb",
                    "Apple iPhone 6 (ATT) 128GB",
                    "Apple iPhone Apple iPhone 6 128GB 412 2 cell 2891",
                    "iPhone 6 ATT 128 GB"
                ])

y_train_text = [['1'],['1'],['1'],['1'],['1'],['1'],['2'],['2'],['2'],['2'],['2'],['2'],['2'],["new york"],["new york"],["new york"],["new york"],["new york"],
                ["new york"],["london"],["london"],["london"],["london"],
                ["london"],["london"],["new york","london"],["new york","london"],
                ["mumbai"],["mumbai"],["mumbai","india"],["mumbai"],
                ["mumbai","new york"],['3'],['3'],['3'],['3'],['3'],['3'],['3'],
                ['4'],['4'],['4'],['4'],['4'],['4'],
                ['5'],['5'],['5'],['5'],['5'],['5'],
                ['6'],['6'],['6'],['6'],['6'],['6']
                ]

X_test = np.array([
                   "Apple 6 16gb Unlocked", 
                   "Apple 6 128gb ATT",
                   'nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too',
                   'hello welcome to mumbai. enjoy it here and mumbai too',
                   'mumbai is better than new york',
                   'Pune is India',
                   'Apple 6 16gb ATT',
                   "Apple 6 16gb Verizon",
                   "apple iphone 6 tmobile 16gb",
                   "iPhone 6 Sprint 16 GB"])
target_names = ['New York', 'London','Mumbai','India','1','2','3','4','5','6']


#lb = preprocessing.LabelBinarizer()
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)


predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))

print all_labels
#print accuracy_score(y_true, y_pred)
print (classifier.score(X_train,Y))    