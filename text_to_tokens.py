"""
Extracting a vocabulary from document collections and generating a 
CountVectorizer model.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
import tempfile
import os

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("CountVectorizerExample")\
        .getOrCreate()

    # Creating simple dataframe
    df = spark.createDataFrame([
        (0, "apple bear chocolat".split(" ")),
        (1, "apple bear bear chocolat apple".split(" ")),
        (2, "apple box".split(" "))
    ], ["id", "words"])
    df.show(truncate=False)

    # fit a CountVectorizerModel from the 'corpus'.
    # vocabSize: maximum size of vocabulary
    # minDF: minimum number of differnt document term must appear in before it
    #        gets included in the vocabulary.
    cv = CountVectorizer(inputCol="words", outputCol="features",
                         vocabSize=3, minDF=2.0, minTF=1.0)
    model = cv.fit(df)
    result = model.transform(df)

    # Printing the result
    result.show(truncate=False)

    print(sorted(model.vocabulary))
    # 'box' did not get included in the vocabulary because it only appeared
    # in 1 document

    # Save the CountVectorizer in a temporary directory.
    tempdir = tempfile.NamedTemporaryFile(delete=False).name
    os.unlink(tempdir)
    cv.save(tempdir)

    # Load the CountVectorizer back into memory
    loadedCv = CountVectorizer.load(tempdir)

    # Verifying loaded CountVectorizer object
    print(loadedCv.getMinDF() == cv.getMinDF(),
          loadedCv.getMinTF() == cv.getMinTF(),
          loadedCv.getVocabSize() == cv.getVocabSize())

    spark.stop()
