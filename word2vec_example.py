"""
Word2Vec trains a model of Map(String, Vector), i.e. transforms a word into a 
code for further natural language processing or machine learning process.
"""

from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Word2VecExample")\
        .getOrCreate()

    # Example 1
    sent = ("a b " * 100 + "a c " * 10).split(" ")
    doc = spark.createDataFrame([(sent,), (sent,)], ["sentence"])
    doc.show(truncate=False)
    word2Vec = Word2Vec(vectorSize=5, seed=42, inputCol="sentence",
                             outputCol="model")
    model = word2Vec.fit(doc)
    model.getVectors().show(truncate=False)

    # Example 2
    # Input data: Each row is a bag of words from a sentence or document.
    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "), ),
        ("I wish Java could use case classes".split(" "), ),
        ("Logistic regression models are neat".split(" "), )
    ], ["text"])

    documentDF.show(truncate=False)

    # Learn a mapping from words to Vectors.
    word2Vec = Word2Vec(vectorSize=4, minCount=0, inputCol="text",
                        outputCol="result")
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

    spark.stop()
