"""from pyspark import SparkContext 
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

parsed_data = [LabeledPoint(0.0, [4.6,3.6,1.0,0.2]),
                LabeledPoint(0.0, [5.7,4.4,1.5,0.4]),
                LabeledPoint(1.0, [6.7,3.1,4.4,1.4]),
                LabeledPoint(0.0, [4.8,3.4,1.6,0.2]),
                LabeledPoint(2.0, [4.4,3.2,1.3,0.2])]     

sc=SparkContext("local","dd")
#model = LogisticRegressionWithSGD.train(sc.parallelize(parsed_data)) # gives error:
 # org.apache.spark.SparkException: Input validation failed.

#model = LogisticRegressionWithLBFGS.train(sc.parallelize(parsed_data), numClasses=3)  # works OK
"""

from pyspark.ml.feature import PCA
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext 
from pyspark.sql import SQLContext


sc=SparkContext("local","dd")
sqlContext = SQLContext(sc)

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
  (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
  (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = sqlContext.createDataFrame(data,["features"])
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
