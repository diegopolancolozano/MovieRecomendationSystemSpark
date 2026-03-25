from pathlib import Path
import os

from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

from spark_worker import assemble_and_scale_features, build_user_genre_features


def _is_remote_path(path: str) -> bool:
	return "://" in path


def get_input_base_path() -> str:
	configured = os.getenv("MOVIELENS_INPUT_BASE_PATH")
	if configured:
		return configured.rstrip("/\\")

	bucket = os.getenv("MOVIELENS_GCS_BUCKET")
	if bucket:
		bucket = bucket.replace("gs://", "").strip("/\\")
		return f"gs://{bucket}/ml-100k"

	local_base = Path(__file__).resolve().parent / "data" / "ml-100k" / "ml-100k"
	return str(local_base)


def get_output_base_path() -> str:
	configured = os.getenv("MOVIELENS_OUTPUT_BASE_PATH")
	if configured:
		return configured.rstrip("/\\")

	bucket = os.getenv("MOVIELENS_GCS_BUCKET")
	if bucket:
		bucket = bucket.replace("gs://", "").strip("/\\")
		return f"gs://{bucket}/processed"

	local_base = Path(__file__).resolve().parent / "data" / "processed"
	local_base.mkdir(parents=True, exist_ok=True)
	return str(local_base)


def build_spark_session() -> SparkSession:
	master_url = os.getenv("SPARK_MASTER_URL", "spark://127.0.0.1:7077")

	return (
		SparkSession.builder.appName("SparkMovieLensMaster")
		.master(master_url)
		.config("spark.sql.shuffle.partitions", "4")
		.config("spark.default.parallelism", "4")
		.config("spark.dynamicAllocation.enabled", "false")
		.config("spark.executor.instances", "2")
		.config("spark.executor.cores", "1")
		.getOrCreate()
	)


def load_ratings_dataframe(spark: SparkSession):
	base_path = get_input_base_path()
	ratings_path = f"{base_path}/u.data"

	if not _is_remote_path(ratings_path) and not Path(ratings_path).exists():
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
		ratings_path
	)


def load_movies_dataframe(spark: SparkSession):
	base_path = get_input_base_path()
	movies_path = f"{base_path}/u.item"

	if not _is_remote_path(movies_path) and not Path(movies_path).exists():
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

	return (
		spark.read.option("delimiter", "|")
		.option("encoding", "ISO-8859-1")
		.schema(movies_schema)
		.csv(movies_path)
	)


def save_feature_dataframes(df_user_genre_long, df_user_genre_wide, df_vectors, feature_columns):
	output_base_path = get_output_base_path()
	long_output = f"{output_base_path}/user_genre_long_csv"
	wide_output = f"{output_base_path}/user_genre_wide_csv"
	vectors_output = f"{output_base_path}/user_kmeans_vectors_csv"

	df_user_genre_long_ordered = df_user_genre_long.orderBy("userId", "genre")
	df_user_genre_wide_ordered = df_user_genre_wide.orderBy("userId")

	raw_arr = vector_to_array("features_raw")
	scaled_arr = vector_to_array("features_scaled")
	df_vectors_flat = df_vectors.select(
		"userId",
		*[raw_arr.getItem(i).alias(f"raw_{name}") for i, name in enumerate(feature_columns)],
		*[scaled_arr.getItem(i).alias(f"scaled_{name}") for i, name in enumerate(feature_columns)],
	).orderBy("userId")

	df_user_genre_long_ordered.write.mode("overwrite").option("header", "true").csv(long_output)
	df_user_genre_wide_ordered.write.mode("overwrite").option("header", "true").csv(wide_output)
	df_vectors_flat.write.mode("overwrite").option("header", "true").csv(vectors_output)

	print("\nFeatures guardados:")
	print(f"- Formato largo (porcentaje por genero): {long_output}")
	print(f"- Formato ancho normalizado (entrada K-Means): {wide_output}")
	print(f"- Vectores ensamblados y escalados: {vectors_output}")
	print("- Nota: Spark escribe carpetas con part-*.csv, no un unico archivo csv.")


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
		print(f"Spark master configurado: {spark.sparkContext.master}")
		print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")
		print(f"Input base path: {get_input_base_path()}")
		print(f"Output base path: {get_output_base_path()}")

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
