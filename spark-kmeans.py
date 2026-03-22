from pathlib import Path
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, countDistinct, explode, lit, sum as spark_sum, when
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType


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


def load_movies_dataframe(spark: SparkSession):
	base_dir = Path(__file__).resolve().parent
	movies_path = base_dir / "data" / "ml-100k" / "ml-100k" / "u.item"

	if not movies_path.exists():
		raise FileNotFoundError(
			f"No se encontro el archivo de peliculas en: {movies_path}"
		)

	movies_schema = StructType(
		[
			StructField("movieId", IntegerType(), nullable=False),
			StructField("title", StringType(), nullable=True),
			StructField("release_date", StringType(), nullable=True),
			StructField("video_release_date", StringType(), nullable=True),
			StructField("imdb_url", StringType(), nullable=True),
			StructField("unknown", IntegerType(), nullable=False),
			StructField("Action", IntegerType(), nullable=False),
			StructField("Adventure", IntegerType(), nullable=False),
			StructField("Animation", IntegerType(), nullable=False),
			StructField("Children", IntegerType(), nullable=False),
			StructField("Comedy", IntegerType(), nullable=False),
			StructField("Crime", IntegerType(), nullable=False),
			StructField("Documentary", IntegerType(), nullable=False),
			StructField("Drama", IntegerType(), nullable=False),
			StructField("Fantasy", IntegerType(), nullable=False),
			StructField("FilmNoir", IntegerType(), nullable=False),
			StructField("Horror", IntegerType(), nullable=False),
			StructField("Musical", IntegerType(), nullable=False),
			StructField("Mystery", IntegerType(), nullable=False),
			StructField("Romance", IntegerType(), nullable=False),
			StructField("SciFi", IntegerType(), nullable=False),
			StructField("Thriller", IntegerType(), nullable=False),
			StructField("War", IntegerType(), nullable=False),
			StructField("Western", IntegerType(), nullable=False),
		]
	)

	# Solo se usa movieId y banderas de genero para el feature engineering.
	return (
		spark.read.option("delimiter", "|")
		.option("encoding", "ISO-8859-1")
		.schema(movies_schema)
		.csv(str(movies_path))
	)


def build_user_genre_features(df_ratings, df_movies, positive_threshold=3):
	genre_columns = [
		"unknown",
		"Action",
		"Adventure",
		"Animation",
		"Children",
		"Comedy",
		"Crime",
		"Documentary",
		"Drama",
		"Fantasy",
		"FilmNoir",
		"Horror",
		"Musical",
		"Mystery",
		"Romance",
		"SciFi",
		"Thriller",
		"War",
		"Western",
	]

	# Excluye ratings neutrales y negativos; se conservan solo opiniones positivas.
	df_positive = df_ratings.filter(col("rating") > positive_threshold)

	# Convierte cada pelicula en varias filas (movieId, genre) segun sus banderas.
	genre_exprs = [
		when(col(g) == 1, lit(g))
		for g in genre_columns
	]

	df_movie_genres = (
		df_movies
		.select("movieId", explode(array(*genre_exprs)).alias("genre"))
		.filter(col("genre").isNotNull())
	)

	df_user_genre_long = (
		df_positive.join(df_movie_genres, on="movieId", how="inner")
		.groupBy("userId", "genre")
		.agg(spark_sum("rating").alias("rating_sum"))
	)

	df_user_genre_wide = (
		df_user_genre_long
		.groupBy("userId")
		.pivot("genre")
		.agg(spark_sum("rating_sum"))
		.fillna(0)
		.orderBy("userId")
	)

	return df_user_genre_long, df_user_genre_wide


def save_feature_dataframes(df_user_genre_long, df_user_genre_wide):
	base_dir = Path(__file__).resolve().parent
	output_dir = base_dir / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	long_output = output_dir / "user_genre_long.csv"
	wide_output = output_dir / "user_genre_wide.csv"

	df_user_genre_long_ordered = df_user_genre_long.orderBy("userId", "genre")
	df_user_genre_wide_ordered = df_user_genre_wide.orderBy("userId")

	with open(long_output, "w", newline="", encoding="utf-8") as f_long:
		writer = csv.writer(f_long)
		writer.writerow(df_user_genre_long_ordered.columns)
		for row in df_user_genre_long_ordered.toLocalIterator():
			writer.writerow([row[c] for c in df_user_genre_long_ordered.columns])

	with open(wide_output, "w", newline="", encoding="utf-8") as f_wide:
		writer = csv.writer(f_wide)
		writer.writerow(df_user_genre_wide_ordered.columns)
		for row in df_user_genre_wide_ordered.toLocalIterator():
			writer.writerow([row[c] for c in df_user_genre_wide_ordered.columns])

	print("\nFeatures guardados:")
	print(f"- Formato largo: {long_output}")
	print(f"- Formato ancho (K-Means): {wide_output}")


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
		df_movies = load_movies_dataframe(spark)

		print("Esquema del DataFrame de ratings:")
		df_ratings.printSchema()

		print("\nMuestra de datos (primeras 5 filas):")
		df_ratings.show(5, truncate=False)

		exploratory_analysis(df_ratings)

		df_user_genre_long, df_user_genre_wide = build_user_genre_features(
			df_ratings, df_movies, positive_threshold=3
		)
		save_feature_dataframes(df_user_genre_long, df_user_genre_wide)
	finally:
		spark.stop()


if __name__ == "__main__":
	main()