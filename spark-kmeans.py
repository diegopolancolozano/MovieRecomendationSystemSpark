from pathlib import Path
import csv

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
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
	genre_count_col = sum(col(g) for g in genre_columns)

	df_movie_genres = (
		df_movies
		.select(
			"movieId",
			genre_count_col.alias("genre_count"),
			explode(array(*genre_exprs)).alias("genre"),
		)
		.filter(col("genre").isNotNull())
	)

	df_user_genre_long = (
		df_positive.join(df_movie_genres, on="movieId", how="inner")
		.withColumn("weighted_rating", col("rating") / col("genre_count"))
		.groupBy("userId", "genre")
		.agg(spark_sum("weighted_rating").alias("rating_sum"))
	)

	df_user_total = (
		df_user_genre_long
		.groupBy("userId")
		.agg(spark_sum("rating_sum").alias("total_positive_rating"))
	)

	df_user_genre_normalized_long = (
		df_user_genre_long.join(df_user_total, on="userId", how="inner")
		.withColumn(
			"rating_pct",
			col("rating_sum") / col("total_positive_rating"),
		)
		.select("userId", "genre", "rating_sum", "rating_pct")
	)

	df_user_genre_wide = (
		df_user_genre_normalized_long
		.groupBy("userId")
		.pivot("genre")
		.agg(spark_sum("rating_pct"))
		.fillna(0)
		.orderBy("userId")
	)

	return df_user_genre_normalized_long, df_user_genre_wide


def assemble_and_scale_features(df_user_genre_wide):
	feature_columns = sorted([c for c in df_user_genre_wide.columns if c != "userId"])

	assembler = VectorAssembler(
		inputCols=feature_columns,
		outputCol="features_raw",
	)
	df_assembled = assembler.transform(df_user_genre_wide)

	scaler = StandardScaler(
		inputCol="features_raw",
		outputCol="features_scaled",
		withStd=True,
		withMean=True,
	)
	scaler_model = scaler.fit(df_assembled)
	df_scaled = scaler_model.transform(df_assembled)

	return df_scaled.select("userId", "features_raw", "features_scaled"), feature_columns


def save_feature_dataframes(df_user_genre_long, df_user_genre_wide, df_vectors, feature_columns):
	base_dir = Path(__file__).resolve().parent
	output_dir = base_dir / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	long_output = output_dir / "user_genre_long.csv"
	wide_output = output_dir / "user_genre_wide.csv"
	vectors_output = output_dir / "user_kmeans_vectors.csv"

	df_user_genre_long_ordered = df_user_genre_long.orderBy("userId", "genre")
	df_user_genre_wide_ordered = df_user_genre_wide.orderBy("userId")
	df_vectors_flat = (
		df_vectors
		.select(
			"userId",
			vector_to_array("features_raw").alias("features_raw_arr"),
			vector_to_array("features_scaled").alias("features_scaled_arr"),
		)
		.orderBy("userId")
	)

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

	with open(vectors_output, "w", newline="", encoding="utf-8") as f_vectors:
		writer = csv.writer(f_vectors)
		raw_headers = [f"raw_{c}" for c in feature_columns]
		scaled_headers = [f"scaled_{c}" for c in feature_columns]
		writer.writerow(["userId", *raw_headers, *scaled_headers])
		for row in df_vectors_flat.toLocalIterator():
			writer.writerow(
				[
					row["userId"],
					*row["features_raw_arr"],
					*row["features_scaled_arr"],
				]
			)

	print("\nFeatures guardados:")
	print(f"- Formato largo (porcentaje por genero): {long_output}")
	print(f"- Formato ancho normalizado (entrada K-Means): {wide_output}")
	print(f"- Vectores ensamblados y escalados: {vectors_output}")


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
		df_vectors, feature_columns = assemble_and_scale_features(df_user_genre_wide)
		save_feature_dataframes(
			df_user_genre_long,
			df_user_genre_wide,
			df_vectors,
			feature_columns,
		)
	finally:
		spark.stop()


if __name__ == "__main__":
	main()