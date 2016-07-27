from textblob.classifiers import NaiveBayesClassifier

train = [
	('iPhone 5s - AT&T, 16GB',		'iPhone 5s - AT&T, 16GB, Space Gray'),
	('iPhone 5 - Verizon, 32GB',	'iPhone 5 - Verizon, 32GB, Black'),
	('iPhone 5 - Verizon, 32GB',	'iPhone 5 - Verizon, 32GB, White'),
	('iPhone 5 - Verizon, 16GB',	'iPhone 5 - Verizon, 16GB, Black'),
	('iPhone 5 - Verizon, 16GB',	'iPhone 5 - Verizon, 16GB, White'),
	('iPhone 5 - AT&T, 32GB',		'iPhone 5 - AT&T, 32GB, Black'),
	('iPhone 5 - AT&T, 32GB',		'iPhone 5 - AT&T, 32GB, White'),
	('iPhone 5 - AT&T, 16GB',		'iPhone 5 - AT&T, 16GB, Black'),
	('iPhone 5 - AT&T, 16GB',		'iPhone 5 - AT&T, 16GB, White'),
	('iPhone 5 - Unlocked, 32GB',	'iPhone 5 - Unlocked, 32GB, Black'),
	('iPhone 5 - Unlocked, 32GB',	'iPhone 5 - Unlocked, 32GB, White'),
	('iPhone 5 - Unlocked, 16GB',	'iPhone 5 - Unlocked, 16GB, Black'),
	('iPhone 5 - Unlocked, 16GB',	'iPhone 5 - Unlocked, 16GB, White'),
	('iPhone 5 - Sprint, 32GB',		'iPhone 5 - Sprint, 32GB, Black'),
	('iPhone 5 - Sprint, 32GB',		'iPhone 5 - Sprint, 32GB, White'),
	('iPhone 5 - Sprint, 16GB',		'iPhone 5 - Sprint, 16GB, Black'),
	('iPhone 5 - Sprint, 16GB',		'iPhone 5 - Sprint, 16GB, White'),
	('iPhone 5 - T-Mobile, 32GB',	'iPhone 5 - T-Mobile, 32GB, Black'),
	('iPhone 5 - T-Mobile, 32GB',	'iPhone 5 - T-Mobile, 32GB, White'),
	('iPhone 5 - T-Mobile, 16GB',	'iPhone 5 - T-Mobile, 16GB, Black'),
	('iPhone 5 - T-Mobile, 16GB',	'iPhone 5 - T-Mobile, 16GB, White'),
	('iPhone 5s - Verizon, 32GB',	'iPhone 5s - Verizon, 32GB, Silver'),
	('iPhone 5s - Verizon, 32GB',	'iPhone 5s - Verizon, 32GB, Gold'),
	('iPhone 5s - Verizon, 32GB',	'iPhone 5s - Verizon, 32GB, Space Gray'),
	('iPhone 5s - Verizon, 16GB',	'iPhone 5s - Verizon, 16GB, Silver'),
	('iPhone 5s - Verizon, 16GB',	'iPhone 5s - Verizon, 16GB, Gold'),
	('iPhone 5s - Verizon, 16GB',	'iPhone 5s - Verizon, 16GB, Space Gray')
]
test = [
	('iPhone 5s - AT&T, 32GB',		'iPhone 5s - AT&T, 32GB, Silver'),
	('iPhone 5s - AT&T, 32GB',		'iPhone 5s - AT&T, 32GB, Gold'),
	('iPhone 5s - AT&T, 32GB',		'iPhone 5s - AT&T, 32GB, Space Gray'),
	('iPhone 5s - Unlocked, 32GB',	'iPhone 5s - Unlocked, 32GB, Silver'),
	('iPhone 5s - Unlocked, 32GB',	'iPhone 5s - Unlocked, 32GB, Gold'),
	('iPhone 5s - Unlocked, 32GB',	'iPhone 5s - Unlocked, 32GB, Space Gray'),
	('iPhone 5s - Unlocked, 16GB',	'iPhone 5s - Unlocked, 16GB, Silver'),
	('iPhone 5s - Unlocked, 16GB',	'iPhone 5s - Unlocked, 16GB, Gold'),
	('iPhone 5s - Unlocked, 16GB',	'iPhone 5s - Unlocked, 16GB, Space Gray')
]

cl = NaiveBayesClassifier(train)
print (cl.classify("iPhone 5s - Unlocked, 16GB")) 
print (cl.classify("iPhone 5s - Sprint, 16GB"))

print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features(5)
