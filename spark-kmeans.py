from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# 1. Inicializar Spark
spark = SparkSession.builder.appName("SparkClustering").getOrCreate()