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



X_train = np.array([
                    "iphone 6 att 16gb",
                    "iPhone 6 ATT 16GB",
                    "Apple iPhone 6 (ATT) 16GB",
                    "apple iphone 6 16gb att",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2889",
                    "iPhone 6 ATT 16gb",
                    "iphone 6 verizon 16gb",
                    "iPhone 6 Verizon 16GB",
                    "apple iphone 6 verizon 16gb",
                    "Apple iPhone 6 (Verizon) 16GB",
                    "apple iphone 6 16gb verizon",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2898",
                    "iPhone 6 Verizon 16gb",
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
                    "iPhone 6 tmobile 16GB",
                    "Apple iPhone 6 (tmobile) 16GB",
                    "apple iphone 6 16gb tmobile",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2895",
                    "iPhone 6 tmobile 16gb",
                    "Apple 6 16gb tmobile",
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
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell",
                    "iPhone 6 Plus Factory Unlocked 16gb",
                    "iphone 6 att 128gb",
                    "iPhone 6 ATT 128GB",
                    "apple iphone 6 att 128gb",
                    "Apple iPhone 6 (ATT) 128GB",
                    "Apple iPhone Apple iPhone 6 128GB 412 2 cell",
                    "iPhone 6 ATT 128gb",
                    "iphone 6 att 64gb",
                    "iPhone 6 ATT 64GB",
                    "apple iphone 6 att 64gb",
                    "Apple iPhone 6 (ATT) 64GB",
                    "apple iphone 6 64gb att",
                    "Apple iPhone Apple iPhone 6 64GB 412 2 cell",
                    "iPhone 6 ATT 64gb",
                    "iPhone 6 Verizon 128GB",
                    "apple iphone 6 verizon 128gb",
                    "Apple iPhone 6 (Verizon) 128GB",
                    "apple iphone 6 128gb verizon",
                    "Apple iPhone Apple iPhone 6 128GB 412 2 cell",
                    "iPhone 6 Verizon 128gb",
                    "Apple 6 128gb Verizon",
                    "iphone 6 verizon 64gb",
                    "iPhone 6 Verizon 64GB",
                    "apple iphone 6 verizon 64gb",
                    "Apple iPhone 6 (Verizon) 64GB",
                    "apple iphone 6 64gb verizon",
                    "iPhone 6 Verizon 64gb",
                    "Apple 6 64gb Verizon",
                    "iphone 6 tmobile 128gb",
                    "iPhone 6 T-Mobile 128GB",
                    "apple iphone 6 tmobile 128gb",
                    "Apple iPhone 6 (tmobile) 128GB",
                    "apple iphone 6 128gb tmobile",
                    "Apple iPhone Apple iPhone 6 128GB 412 2 cell",
                    "Apple 6 128gb tmobile",
                    "iphone 6 tmobile 64gb",
                    "iPhone 6 tmobile 64GB",
                    "apple iphone 6 tmobile 64gb",
                    "Apple iPhone 6 tmobile 64GB",
                    "apple iphone 6 64gb tmobile",
                    "Apple iPhone Apple iPhone 6 64GB 412 2 cell 2896",
                    "iPhone 6 tmobile 64gb",
                    "iphone 6 sprint 128gb",
                    "iPhone 6 Sprint 128GB",
                    "Apple iPhone 6 Sprint 128GB",
                    "apple iphone 6 128gb sprint",
                    "Apple iPhone Apple iPhone 6 128GB",
                    "iPhone 6 Sprint 128gb",
                    "iphone 6 sprint 64gb",
                    "iPhone 6 Sprint 64GB",
                    "Apple iPhone 6 Sprint 64GB",
                    "apple iphone 6 64gb sprint",
                    "Apple iPhone Apple iPhone 6 64GB 412 2 cell 2893"
                    "iPhone 6 Sprint 64gb",
                    "iPhone 6 Unlocked 128GB",
                    "apple iphone 6 128gb unlocked",
                    "Apple iPhone Apple iPhone 6 128GB 412 2 cell 2906",
                    "iPhone 6 Plus Factory Unlocked 128gb",
                    "Apple 6 128gb Unlocked",
                    "iphone 6 unlocked 64gb",
                    "iPhone 6 Unlocked, 64GB",
                    "apple iphone 6 64gb unlocked",
                    "Apple iPhone Apple iPhone 6 64GB 412 2 cell 2905",
                    "Apple 6 64gb Unlocked",
                    "iPhone 6 Plus Factory Unlocked 64gb",
                    "iPhone 6 att, 16GB",
                    "apple iphone 6 att 16gb",
                    "iPhone 6 att 16gb",
                    "Apple 6 16gb att",
                    "iPhone 6 att, 16GB",
                    "apple iphone 6 att 16gb",
                    "Apple iPhone 6 (ATT) 16GB",
                    "iPhone 6 att 16gb",
                    "Apple 6 16gb att",
                    "iPhone 6 Verizon, 16GB",
                    "Apple 6 16gb Verizon",
                    "iphone 6 tmobile 16gb",
                    "iPhone 6 tmobile, 16GB",
                    "apple iphone 6 tmobile 16gb",
                    "Apple iPhone 6 (tmobile) 16GB",
                    "apple iphone 6 16gb tmobile",
                    "Apple iPhone Apple iPhone 6 16GB 412 2 cell 2895",
                    "iPhone 6 tmobile 16gb",
                   
                ])
print X_train

y_train_text = [['1','16','17'],['1'],['1'],['1','16','17'],['1','16','17'],['1'],
                ['2','18','19'],['2'],['2','18','19'],['2','18','19'],['2','18','19'],['2','18','19'],['2','18','19'],
                ["new york"],["new york"],["new york"],["new york"],["new york"],
                ["new york"],["london"],["london"],["london"],["london"],
                ["london"],["london"],["new york","london"],["new york","london"],
                ["mumbai"],["mumbai"],["mumbai","india"],["mumbai"],
                ["mumbai","new york"],['3','20'],['3'],['3'],['3'],['3'],['3'],['3','20'],
                ['4'],['4'],['4'],['4'],['4'],['4'],
                ['5'],['5'],['5'],['5'],['5'],['5'],
                ['6'],['6'],['6'],['6'],['6'],['6'],
                ['7'],['7'],['7'],['7'],['7'],['7'],['7'],
                ['8'],['8'],['8'],['8'],['8'],['8'],['8'],
                ['9'],['9'],['9'],['9'],['9'],['9'],['9'],
                ['10'],['10'],['10'],['10'],['10'],['10'],['10'],
                ['11'],['11'],['11'],['11'],['11'],['11'],['11'],
                ['12'],['12'],['12'],['12'],['12'],['12'],
                ['13'],['13'],['13'],['13'],['13'],
                ['14'],['14'],['14'],['14'],['14'],
                ['15'],['15'],['15'],['15'],['15'],['15'],
                ['16'],['16'],['16'],['16'],
                ['17'],['17'],['17'],['17'],['17'],
                ['18','19'],['18','19'],
                ['20'],['20'],['20'],['20'],['20'],['20'],['20']
                ]

X_test = np.array(["Apple 6 16gb Unlocked", 
                   "Apple 6 128gb ATT",
                   'nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   "Apple 6 64gb ATT",
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too',
                   'hello welcome to mumbai. enjoy it here and mumbai too',
                   "iphone 6 verizon 128gb",
                   "Apple iPhone Apple iPhone 6 64GB 412 2 cell 2899",
                   'iPhone 6 tmobile 128gb',
                   'Pune is India',
                   'Apple 6 16gb ATT',
                   "Apple 6 16gb Verizon",
                   "apple iphone 6 tmobile 16gb",
                   "iPhone 6 Sprint 16gb",
                   "Apple iPhone Apple iPhone 6 128GB 412 2 cell",
                   "Apple 6 64gb tmobile",
                   "Apple iPhone Apple iPhone 6 128GB",
                   "apple iphone 6 sprint 128gb",
                   "apple iphone 6 sprint 64gb",
                   "iphone 6 unlocked 128gb",
                   "iPhone 6 Plus Factory Unlocked 16gb",
                   "Apple iPhone 6 (ATT) 16GB",
                   "iphone 6 att 16gb",
                   "iphone 6 verizon 16gb"])

y_test =  [
            ['5'],
            ['6'],
            ['new york'],
            ['london'],
            ['london'],
            ['7'],
            ['new york','london'],
            ['new york','london'],
            ['mumbai'],
            ['8'],
            ['9'],
            ['10'],
            ['India'],
            ['1'],
            ['2','18','19'],
            ['3','20'],
            ['4'],
            ['6','8','10','12'],
            ['11'],
            ['6','8','10','12','14'],
            ['12'],
            ['13'],
            ['14'],
            ['5','14','15'],
            ['1','16','17'],
            ['1','16','17'],
            ['2','18','19']]

target_names = ['New York', 'London','Mumbai','India','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                '16','17','18','19','20'
                ]


#lb = preprocessing.LabelBinarizer()
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)
y_1 = lb.fit_transform(y_test)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(C=10.)))])
    
    #('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)))])
    #('clf', OneVsRestClassifier(LinearSVC(C=10.)))])
    
X_train = [x.lower() for x in X_train]    
classifier.fit(X_train, Y)


X_test = [x.lower() for x in X_test]    
predicted = classifier.predict(X_test)

all_labels = lb.inverse_transform(predicted)

#print all_labels
for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))

# get the accuracy
print X_train
print (classifier.score(X_train,Y))    
print (classifier.score(X_test,y_1))    
print(classification_report(y_1, predicted))
#print (classifier.score(y_test,all_labels))    