from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Word2Vec
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint



sc=SparkContext("local","dd")
sqlContext = SQLContext(sc)

"""sentenceData = sqlContext.createDataFrame([
  (0, "Hi I heard about Spark"),
  (0, "I wish Java could use case classes"),
  (1, "Logistic regression models are neat")
], ["label", "sentence"])
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
for features_label in rescaledData.select("features", "label").take(3):
  print(features_label)"""



# Input data: Each row is a bag of words from a sentence or document.
documentDF = sqlContext.createDataFrame([
  ("Hi I heard about Spark".split(" "), 0),
  ("I wish Java could use case classes".split(" "), 0),
  ("Logistic regression models are neat".split(" "), 1)
], ["text","label"])
# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
result = model.transform(documentDF)

for features_label in result.select("result", "label").take(3):
  print(features_label)


wordsvectors = result["label","result"].map(lambda row: LabeledPoint(row[0], row[1]))
model = SVMWithSGD.train(wordsvectors, iterations=100)

labelsAndPreds = wordsvectors.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(wordsvectors.count())
print("Training Error = " + str(trainErr))

print("*********************************************************")
print(result.take(5))
print(labelsAndPreds.take(5))
print("*********************************************************")

