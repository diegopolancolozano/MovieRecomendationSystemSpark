from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.functions import array, col, explode, lit, sum as spark_sum, when


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

	df_positive = df_ratings.filter(col("rating") > positive_threshold)

	genre_exprs = [when(col(g) == 1, lit(g)) for g in genre_columns]
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
