from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cluster, datasets
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
import numpy as np
#from pyspark.mllib.feature import HashingTF
#from pyspark import SparkContext



"""corpus = [
	(1,
		"iphone 6 att 16gb"
		"iPhone 6 ATT 16GB"
		"Apple iPhone 6 (ATT) 16GB"
		"apple iphone 6 16gb att"
		"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2889"
		"iPhone 6 ATT 16 GB"
		"Apple 6 16gb ATT"
	),
	(2,
		"iphone 6 verizon 16gb"
		"iPhone 6 Verizon 16GB"
		"apple iphone 6 verizon 16gb"
		"Apple iPhone 6 (Verizon) 16GB"
		"apple iphone 6 16gb verizon"
		"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2898"
		"iPhone 6 Verizon 16 GB"
		"Apple 6 16gb Verizon"

	),
	(3,
		"iphone 6 tmobile 16gb"
		"iPhone 6 T Mobile 16GB"
		"apple iphone 6 tmobile 16gb"
		"Apple iPhone 6 (T Mobile) 16GB"
		"apple iphone 6 16gb t mobile"
		"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2895"
		"iPhone 6 T Mobile 16 GB"
		"Apple 6 16gb T Mobile"

	),
	(4,
		"iphone 6 sprint 16gb",
		"iPhone 6 Sprint 16GB",
		"apple iphone 6 sprint 16gb",
		"Apple iPhone 6 (Sprint) 16GB",
		"apple iphone 6 16gb sprint",
		"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2892",
		"iPhone 6 Sprint 16 GB"

	),
	(5,
		"iphone 6 unlocked 16gb"
		"iPhone 6 Unlocked 16GB"
		"Apple iPhone 6 (Factory Unlocked) 16GB"
		"apple iphone 6 16gb unlocked"
		"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2904"
		"iPhone 6 Plus Factory Unlocked 16 GB"
		"Apple 6 16gb Unlocked"
	),
	(6,
		"iphone 6 att 128gb"
		"iPhone 6 ATT 128GB"
		"apple iphone 6 att 128gb"
		"Apple iPhone 6 (ATT) 128GB"
		"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2891"
		"iPhone 6 ATT 128 GB"
		"Apple 6 128gb ATT"

	)
	(7,
		"iphone 6 att 64gb",
		"iPhone 6 ATT 64GB",
		"apple iphone 6 att 64gb",
		"Apple iPhone 6 (ATT) 64GB",
		"apple iphone 6 64gb att"
		"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2890",
		"iPhone 6 ATT 64 GB",
		"Apple 6 64gb ATT"
	),
	(8,
		"iphone 6 verizon 128gb",
		"iPhone 6 Verizon 128GB",
		"apple iphone 6 verizon 128gb",
		"Apple iPhone 6 (Verizon) 128GB",
		"apple iphone 6 128gb verizon",
		"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2900",
		"iPhone 6 Verizon 128 GB",
		"Apple 6 128gb Verizon",
	),	
	(9,
		"iphone 6 verizon 64gb",
		"iPhone 6 Verizon 64GB"
		"apple iphone 6 verizon 64gb",
		"Apple iPhone 6 (Verizon) 64GB",
		"apple iphone 6 64gb verizon",
		"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2899",
		"iPhone 6 Verizon 64 GB",
		"Apple 6 64gb Verizon"

	),
	(10,
		"iphone 6 tmobile 128gb",
		"iPhone 6 T Mobile 128GB",
		"apple iphone 6 tmobile 128gb",
		"Apple iPhone 6 (T Mobile) 128GB",
		"apple iphone 6 128gb t mobile",
		"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2897",
		"iPhone 6 T Mobile 128 GB",
		"Apple 6 128gb T Mobile"
	),
	(11,
		"iphone 6 tmobile 64gb",
		"iPhone 6 T Mobile 64GB",
		"apple iphone 6 tmobile 64gb",
		"Apple iPhone 6 T Mobile 64GB",
		"apple iphone 6 64gb t mobile",
		"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2896",
		"iPhone 6 T Mobile 64 GB",
		"Apple 6 64gb T Mobile"

	),
	(12,
		"iphone 6 sprint 128gb",
		"iPhone 6 Sprint 128GB",
		"apple iphone 6 sprint 128gb",
		"Apple iPhone 6 Sprint 128GB",
		"apple iphone 6 128gb sprint",
		"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2894",
		"iPhone 6 Sprint 128 GB"


	),
	(13,
		"iphone 6 sprint 64gb",
		"iPhone 6 Sprint 64GB",
		"apple iphone 6 sprint 64gb",
		"Apple iPhone 6 Sprint 64GB",
		"apple iphone 6 64gb sprint",
		"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2893"
		"iPhone 6 Sprint 64 GB"
	),
	(14,
		"iphone 6 unlocked 128gb",
		"iPhone 6 Unlocked 128GB",
		"apple iphone 6 128gb unlocked",
		"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2906",
		"iPhone 6 Plus Factory Unlocked 128 GB",
		"Apple 6 128gb Unlocked"),
	),
	(15,
		"iphone 6 unlocked 64gb",
		"iPhone 6 Unlocked, 64GB",
		"apple iphone 6 64gb unlocked",
		"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2905",
		"iPhone 6 Plus Factory Unlocked 64 GB",
		"Apple 6 64gb Unlocked"
	)	

	
	
	
]"""


corpus = [
	(1,"iphone 6 att 16gb"),
	(1,"iPhone 6 ATT 16GB"),
	(1,"Apple iPhone 6 (ATT) 16GB"),
	(1,"apple iphone 6 16gb att"),
	(1,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2889"),
	(1,"iPhone 6 ATT 16 GB"),
	(1,"Apple 6 16gb ATT"),
	(2,"iphone 6 verizon 16gb"),
	(2,"iPhone 6 Verizon 16GB"),
	(2,"apple iphone 6 verizon 16gb"),
	(2,"Apple iPhone 6 (Verizon) 16GB"),
	(2,"apple iphone 6 16gb verizon"),
	(2,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2898"),
	(2,"iPhone 6 Verizon 16 GB"),
	(2,"Apple 6 16gb Verizon"),
	(3,"iphone 6 tmobile 16gb"),
	(3,"iPhone 6 T Mobile 16GB"),
	(3,"apple iphone 6 tmobile 16gb"),
	(3,"Apple iPhone 6 (T Mobile) 16GB"),
	(3,"apple iphone 6 16gb t mobile"),
	(3,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2895"),
	(3,"iPhone 6 T Mobile 16 GB"),
	(3,"Apple 6 16gb T Mobile"),
	(4,"iphone 6 sprint 16gb"),
	(4,"iPhone 6 Sprint 16GB"),
	(4,"apple iphone 6 sprint 16gb"),
	(4,"Apple iPhone 6 (Sprint) 16GB"),
	(4,"apple iphone 6 16gb sprint"),
	(4,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2892"),
	(4,"iPhone 6 Sprint 16 GB"),
	(5,"iphone 6 unlocked 16gb"),
	(5,"iPhone 6 Unlocked 16GB"),
	(5,"Apple iPhone 6 (Factory Unlocked) 16GB"),
	(5,"apple iphone 6 16gb unlocked"),
	(5,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2904"),
	(5,"iPhone 6 Plus Factory Unlocked 16 GB"),
	(5,"Apple 6 16gb Unlocked"),
	(6,"iphone 6 att 128gb"),
	(6,"iPhone 6 ATT 128GB"),
	(6,"apple iphone 6 att 128gb"),
	(6,"Apple iPhone 6 (ATT) 128GB"),
	(6,"apple iphone 6 128gb att"),
	(6,"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2891"),
	(6,"iPhone 6 ATT 128 GB"),
	(6,"Apple 6 128gb ATT"),
	(7,"iphone 6 att 64gb"),
	(7,"iPhone 6 ATT 64GB"),
	(7,"apple iphone 6 att 64gb"),
	(7,"Apple iPhone 6 (ATT) 64GB"),
	(7,"apple iphone 6 64gb att"),
	(7,"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2890"),
	(7,"iPhone 6 ATT 64 GB"),
	(7,"Apple 6 64gb ATT"),
	(8,"iphone 6 verizon 128gb"),
	(8,"iPhone 6 Verizon 128GB"),
	(8,"apple iphone 6 verizon 128gb"),
	(8,"Apple iPhone 6 (Verizon) 128GB"),
	(8,"apple iphone 6 128gb verizon"),
	(8,"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2900"),
	(8,"iPhone 6 Verizon 128 GB"),
	(8,"Apple 6 128gb Verizon"),
	(9,"iphone 6 verizon 64gb"),
	(9,"iPhone 6 Verizon 64GB"),
	(9,"apple iphone 6 verizon 64gb"),
	(9,"Apple iPhone 6 (Verizon) 64GB"),
	(9,"apple iphone 6 64gb verizon"),
	(9,"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2899"),
	(9,"iPhone 6 Verizon 64 GB"),
	(9,"Apple 6 64gb Verizon"),
	(10,"iphone 6 tmobile 128gb"),
	(10,"iPhone 6 T Mobile 128GB"),
	(10,"apple iphone 6 tmobile 128gb"),
	(10,"Apple iPhone 6 (T Mobile) 128GB"),
	(10,"apple iphone 6 128gb t mobile"),
	(10,"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2897"),
	(10,"iPhone 6 T Mobile 128 GB"),
	(10,"Apple 6 128gb T Mobile"),
	(11,"iphone 6 tmobile 64gb"),
	(11,"iPhone 6 T Mobile 64GB"),
	(11,"apple iphone 6 tmobile 64gb"),
	(11,"Apple iPhone 6 (T Mobile) 64GB"),
	(11,"apple iphone 6 64gb t mobile"),
	(11,"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2896"),
	(11,"iPhone 6 T Mobile 64 GB"),
	(11,"Apple 6 64gb T Mobile"),
	(12,"iphone 6 sprint 128gb"),
	(12,"iPhone 6 Sprint 128GB"),
	(12,"apple iphone 6 sprint 128gb"),
	(12,"Apple iPhone 6 (Sprint) 128GB"),
	(12,"apple iphone 6 128gb sprint"),
	(12,"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2894"),
	(12,"iPhone 6 Sprint 128 GB"),
	(13,"iphone 6 sprint 64gb"),
	(13,"iPhone 6 Sprint 64GB"),
	(13,"apple iphone 6 sprint 64gb"),
	(13,"Apple iPhone 6 (Sprint) 64GB"),
	(13,"apple iphone 6 64gb sprint"),
	(13,"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2893"),
	(13,"iPhone 6 Sprint 64 GB"),
	(14,"iphone 6 unlocked 128gb"),
	(14,"iPhone 6 Unlocked 128GB"),
	(14,"apple iphone 6 128gb unlocked"),
	(14,"364 31"),
	(14,"Apple iPhone Apple iPhone 6 128GB 412 2 cell 2906"),
	(14,"iPhone 6 Plus Factory Unlocked 128 GB"),
	(14,"Apple 6 128gb Unlocked"),
	(15,"iphone 6 unlocked 64gb"),
	(15,"iPhone 6 Unlocked, 64GB"),
	(15,"apple iphone 6 64gb unlocked"),
	(15,"Apple iPhone Apple iPhone 6 64GB 412 2 cell 2905"),
	(15,"iPhone 6 Plus Factory Unlocked 64 GB"),
	(15,"Apple 6 64gb Unlocked"),
	(16,"iphone 6 att 16gb"),
	(16,"iPhone 6 AT&T, 16GB"),
	(16,"apple iphone 6 att 16gb"),
	(16,"Apple iPhone 6 (ATT) 16GB"),
	(16,"apple iphone 6 16gb att"),
	(16,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2889"),
	(16,"iPhone 6 AT&T 16 GB"),
	(16,"Apple 6 16gb AT&T"),
	(17,"iphone 6 att 16gb"),
	(17,"iPhone 6 AT&T, 16GB"),
	(17,"apple iphone 6 att 16gb"),
	(17,"Apple iPhone 6 (ATT) 16GB"),
	(17,"apple iphone 6 16gb att"),
	(17,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2889"),
	(17,"iPhone 6 AT&T 16 GB"),
	(17,"Apple 6 16gb AT&T"),
	(18,"iphone 6 verizon 16gb"),
	(18,"iPhone 6 Verizon, 16GB"),
	(18,"apple iphone 6 verizon 16gb"),
	(18,"Apple iPhone 6 (Verizon) 16GB"),
	(18,"apple iphone 6 16gb verizon"),
	(18,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2898"),
	(18,"iPhone 6 Verizon 16 GB"),
	(18,"Apple 6 16gb Verizon"),
	(19,"iphone 6 verizon 16gb"),
	(19,"iPhone 6 Verizon, 16GB"),
	(19,"apple iphone 6 verizon 16gb"),
	(19,"Apple iPhone 6 (Verizon) 16GB"),
	(19,"apple iphone 6 16gb verizon"),
	(19,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2898"),
	(19,"iPhone 6 Verizon 16 GB"),
	(19,"Apple 6 16gb Verizon"),
	(20,"iphone 6 tmobile 16gb"),
	(20,"iPhone 6 T Mobile, 16GB"),
	(20,"apple iphone 6 tmobile 16gb"),
	(20,"Apple iPhone 6 (T Mobile) 16GB"),
	(20,"apple iphone 6 16gb t mobile"),
	(20,"Apple iPhone Apple iPhone 6 16GB 412 2 cell 2895"),
	(20,"iPhone 6 T Mobile 16 GB"),
	(20,"Apple 6 16gb T Mobile")
]
print len(corpus)

def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TFIDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]
    
    # Create the document corpus list
    corpus1 = [d[1].lower() for d in docs]

    # Create the TFIDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus1)
    #hashingTF = HashingTF()
    #X = hashingTF.transform(corpus)
    return X, y


#sc = SparkContext()

# Vectorise and TFIDF transform the corpus 
X, y = create_tfidf_training_data(corpus)

# Create the trainingtestf the data
X_train, X_test, y_train, y_test =  train_test_split(
        X, y, test_size=0.2, random_state=42
)



def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = OneVsRestClassifier(SVC(C=1000000.0, gamma='auto', kernel='rbf'))
    svm.fit(X, y)
    return svm

# Create and train the Support Vector Machine
svm = train_svm(X_train, y_train)   


# Make an array of predictions on the test set
pred = svm.predict(X_test)

# Output the hitrate and the confusion matrix for each model
print("SVM: ")
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))
#print(confusion_matrix(pred, y_test)) 


svm2 = OneVsOneClassifier(LinearSVC(C=100.))
svm2.fit(X_train, y_train)

# Make an array of predictions on the test set
pred = svm2.predict(X_test)

# Output the hitrate and the confusion matrix for each model
print("LinearSVC: ")
print(svm2.score(X_train, y_train))
print(svm2.score(X_test, y_test))


from sklearn.neighbors import KNeighborsClassifier
neigh = (KNeighborsClassifier(n_neighbors=2))
neigh.fit(X_train, y_train) 
pred = neigh.predict(X_test)
print("knn: ")
print(neigh.score(X_train,y_train))
print(neigh.score(X_test,y_test))


from sklearn.ensemble import RandomForestClassifier
clf = (RandomForestClassifier(n_estimators=5,max_depth=None,min_samples_split=5, random_state=15))
clf = clf.fit(X_train, y_train)

# Make an array of predictions on the test set
pred1 = clf.predict(X_test)

# Output the hitrate and the confusion matrix for each model
print("RandomForestClassifier:")
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
#print(confusion_matrix(pred1, y_test))


from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([ ('vect', CountVectorizer()),
	             ('tfidf', TfidfTransformer()),
	             ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))            
#clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
#clf = clf.fit(X_train,y_train)
# Make an array of predictions on the test set
#pred1 = clf.predict(X_test)

# Output the hitrate and the confusion matrix for each model
#print("SGD:")
#print(clf.score(X_train,y_train))
#print(clf.score(X_test,y_test))
#print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
#print(confusion_matrix(pred1, y_test))
