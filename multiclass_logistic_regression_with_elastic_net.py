"""
Multiclass Logistic Regression With Elastic Net.

Multiclass classification is supported via multinomial logistic (softmax) 
regression. In multinomial logistic regression, the algorithm produces K sets 
of coefficients, or a matrix of dimension K×J where K is the number of 
outcome classes and J is the number of features. If the algorithm is fit with 
an intercept term then a length K vector of intercepts is available.

Multinomial coefficients are available as coefficientMatrix and intercepts are 
available as interceptVector.

coefficients and intercept methods on a logistic regression model trained with 
multinomial family are not supported. Use coefficientMatrix and interceptVector 
instead.
"""

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("MulticlassLogisticRegressionWithElasticNet") \
        .getOrCreate()

    # Load training data
    training = spark \
        .read \
        .format("libsvm") \
        .load("data/sample_multiclass_classification_data.txt")

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for multinomial logistic regression
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))

    spark.stop()
