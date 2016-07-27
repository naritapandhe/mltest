#Read a CSV File in PythonPython

import csv
corpus = []

"""with open('gadget_vendors_iden-SHEET-3.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, dialect=csv.excel_tab)
	for row in reader:
		#print ', '.join(row)
		corpus.append(row)

print corpus"""

"""corpus = [
	{ 'name':"iPhone 5s - AT&T, 16GB",'class':"iPhone 5s - AT&T, 16GB, Space Gray"},
	{ 'name':"iPhone 5 - Verizon, 32GB",'class':"iPhone 5 - Verizon, 32GB, Black"},
	{ 'name':"iPhone 5 - Verizon, 32GB",'class':"iPhone 5 - Verizon, 32GB, White"},
	{ 'name':"iPhone 5 - Verizon, 16GB",'class':"iPhone 5 - Verizon, 16GB, Black"},
	{ 'name':"iPhone 5 - Verizon, 16GB",'class':"iPhone 5 - Verizon, 16GB, White"},
	{ 'name':"iPhone 5 - AT&T, 32GB",'class':"iPhone 5 - AT&T, 32GB, Black"},
	{ 'name':"iPhone 5 - AT&T, 32GB",'class':"iPhone 5 - AT&T, 32GB, White"},
	{ 'name':"iPhone 5 - AT&T, 16GB",'class':"iPhone 5 - AT&T, 16GB, Black"}

]"""
"""measurements = [
     {'city': 'Dubai', 'temperature': 33.},
     {'city': 'London', 'temperature': 12.},
     {'city': 'San Fransisco', 'temperature': 18.},
]


"""

"""X = 
y = [0, 0, 0, 1]
sample = 'iPhone 5 - AT&T, 16GB';

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
XArray=vec.fit_transform(X).toArray()
YArray=vec.fit_transform(y)
SampleArray=vec.fit_transform(sample)


print(neigh.predict(SampleArray))"""

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)

corpusX = [ ["iPhone 5s - AT&T, 16GB"], 
			["iPhone 5 - Verizon, 32GB"],
			["iPhone 5 - Verizon, 32GB"],
			["iPhone 5 - Verizon, 16GB"]
		  ]	
corpusY = [
			'iPhone 5s - AT&T, 16GB, ',
			'iPhone 5 - Verizon, 32GB',
			'iPhone 5 - Verizon, 32GB',
			'iPhone 5 - Verizon, 16GB'
		  ]
corpusZ = [['iPhone 5 - Verizon, 16GB']]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer=lambda d: d[0].split(', ')).fit(corpusX)
x=tfidf.fit_transform(corpusX)
print x

tfidf2 = TfidfVectorizer(analyzer=lambda d: d[0].split(', ')).fit(corpusY)
y=tfidf.fit_transform(corpusY)
print y

tfidf2 = TfidfVectorizer(analyzer=lambda d: d[0].split(', ')).fit(corpusZ)
z=tfidf.fit_transform(corpusZ)
print y


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(x, y) 
neigh.predict([[z]])

"""xx = vectorizer.fit_transform(x)
Z = vectorizer.fit_transform(corpusZ)

print "X:"
print  X



neigh.predict(Z)"""