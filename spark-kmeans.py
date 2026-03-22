from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import IntegerType, LongType, StructField, StructType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline


def build_spark_session() -> SparkSession:
	return (
		SparkSession.builder.appName("SparkMovieLensEDA")
		.config("spark.sql.shuffle.partitions", "4")
		.getOrCreate()
	)


def load_ratings_dataframe(spark: SparkSession):
	base_dir = Path(__file__).resolve().parent
	ratings_path = base_dir / "data" / "ml-100k" / "ml-100k" / "u.data"

	if not ratings_path.exists():
		raise FileNotFoundError(
			f"No se encontro el archivo de ratings en: {ratings_path}"
		)

	ratings_schema = StructType(
		[
			StructField("userId", IntegerType(), nullable=False),
			StructField("movieId", IntegerType(), nullable=False),
			StructField("rating", IntegerType(), nullable=False),
			StructField("timestamp", LongType(), nullable=False),
		]
	)

	return spark.read.option("delimiter", "\t").schema(ratings_schema).csv(
		str(ratings_path)
	)


def exploratory_analysis(df_ratings):
	total_records = df_ratings.count()
	total_users = df_ratings.select(countDistinct("userId").alias("users")).first()[
		"users"
	]
	total_movies = df_ratings.select(
		countDistinct("movieId").alias("movies")
	).first()["movies"]

	rating_distribution = df_ratings.groupBy("rating").count().orderBy("rating")

	print("\n=== ANALISIS EXPLORATORIO MOVIELENS 100K ===")
	print(f"Total de registros de ratings: {total_records}")
	print(f"Numero de usuarios unicos: {total_users}")
	print(f"Numero de peliculas unicas: {total_movies}")

	print("\nDistribucion de ratings:")
	rating_distribution.show(truncate=False)


def main():
	spark = build_spark_session()

	try:
		df_ratings = load_ratings_dataframe(spark)

		print("Esquema del DataFrame de ratings:")
		df_ratings.printSchema()

		print("\nMuestra de datos (primeras 5 filas):")
		df_ratings.show(5, truncate=False)

		exploratory_analysis(df_ratings)
	finally:
		spark.stop()


if __name__ == "__main__":
	main()