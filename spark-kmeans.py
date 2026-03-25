import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, explode, array, lit, when, sum as spark_sum, avg
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def build_spark_session() -> SparkSession:
    master_url = os.getenv("SPARK_MASTER_URL", "spark://127.0.0.1:7077")
    
    return (
        SparkSession.builder
        .appName("MovieLens_KMeans_Clustering")
        .master(master_url)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )


def get_bucket_path() -> str:
    bucket = os.getenv("MOVIELENS_GCS_BUCKET")
    if bucket:
        bucket = bucket.replace("gs://", "").strip("/\\")
        return f"gs://{bucket}/ml-100k"
    return "data/ml-100k/ml-100k"


def get_output_path() -> str:
    bucket = os.getenv("MOVIELENS_GCS_BUCKET")
    if bucket:
        bucket = bucket.replace("gs://", "").strip("/\\")
        return f"gs://{bucket}/output"
    return "data/output"


def load_data(spark: SparkSession, bucket_path: str):
    ratings_schema = StructType([
        StructField("userId", IntegerType(), nullable=False),
        StructField("movieId", IntegerType(), nullable=False),
        StructField("rating", IntegerType(), nullable=False),
        StructField("timestamp", LongType(), nullable=False),
    ])

    movies_schema = StructType([
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
    ])

    df_ratings = spark.read.option("delimiter", "\t").schema(ratings_schema).csv(f"{bucket_path}/u.data")
    df_movies = spark.read.option("delimiter", "|").option("encoding", "ISO-8859-1").schema(movies_schema).csv(f"{bucket_path}/u.item")
    
    return df_ratings, df_movies


def exploratory_analysis(df_ratings):
    print("\n=== ANALISIS EXPLORATORIO MOVIELENS 100K ===")
    
    total_users = df_ratings.select(countDistinct("userId")).first()[0]
    total_movies = df_ratings.select(countDistinct("movieId")).first()[0]
    total_ratings = df_ratings.count()
    
    print(f"Número de usuarios únicos: {total_users}")
    print(f"Número de películas únicas: {total_movies}")
    print(f"Total de ratings: {total_ratings}")
    
    print("\nDistribución de ratings:")
    df_ratings.groupBy("rating").count().orderBy("rating").show()
    
    avg_rating = df_ratings.select(avg("rating")).first()[0]
    print(f"\nRating promedio: {avg_rating:.2f}")


def build_features(df_ratings, df_movies):
    genre_columns = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical",
        "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
    ]

    # Filtrar ratings positivos (>3)
    df_positive = df_ratings.filter(col("rating") > 3)

    # Preparar géneros
    genre_exprs = [when(col(g) == 1, lit(g)) for g in genre_columns]
    genre_count_col = sum(col(g) for g in genre_columns)

    df_movie_genres = df_movies.select(
        "movieId",
        genre_count_col.alias("genre_count"),
        explode(array(*genre_exprs)).alias("genre")
    ).filter(col("genre").isNotNull())

    # Calcular weighted rating por género
    df_user_genre = df_positive.join(df_movie_genres, on="movieId") \
        .withColumn("weighted_rating", col("rating") / col("genre_count")) \
        .groupBy("userId").pivot("genre").agg(spark_sum("weighted_rating")).fillna(0)

    print(f"\n=== FEATURES CONSTRUIDOS ===")
    print(f"Usuarios con features: {df_user_genre.count()}")
    print(f"Géneros (features): {len(genre_columns)}")
    
    return df_user_genre, genre_columns


def apply_kmeans(spark, df_features, genre_columns, output_path):
    input_cols = [c for c in df_features.columns if c != "userId"]
    
    # Ensamblar y escalar features
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features")
    df_assembled = assembler.transform(df_features)
    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    print("\n=== APLICANDO K-MEANS ===")
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    
    best_k = None
    best_score = -1
    best_predictions = None
    
    for k in [3, 5, 8]:
        kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=20)
        model = kmeans.fit(df_scaled)
        predictions = model.transform(df_scaled)
        
        silhouette = evaluator.evaluate(predictions)
        print(f"K={k} -> Silhouette Score: {silhouette:.4f}")
        
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
            best_predictions = predictions
    
    print(f"\nMejor K: {best_k} con Silhouette Score: {best_score:.4f}")
    
    # Guardar resultados del mejor modelo
    best_predictions.select("userId", "prediction").write.mode("overwrite").csv(f"{output_path}/clusters_k{best_k}")
    print(f"\nClusters guardados en: {output_path}/clusters_k{best_k}")
    
    return best_predictions, best_k, genre_columns


def analyze_clusters(df_predictions, k, df_features, genre_columns):
    print(f"\n=== ANALISIS DE CLUSTERS (K={k}) ===")
    
    # Distribución de usuarios por cluster
    cluster_dist = df_predictions.groupBy("prediction").count().orderBy("prediction")
    print("\nDistribución de usuarios por cluster:")
    cluster_dist.show()
    
    # Unir con features originales para análisis
    df_analysis = df_predictions.select("userId", "prediction").join(df_features, on="userId")
    
    # Calcular promedios por cluster para cada género
    print("\nCaracterísticas promedio por cluster (top géneros):")
    for cluster_id in range(k):
        df_cluster = df_analysis.filter(col("prediction") == cluster_id)
        
        # Calcular promedio de cada género
        genre_avgs = []
        for genre in genre_columns:
            avg_val = df_cluster.select(avg(genre)).first()[0]
            genre_avgs.append((genre, avg_val))
        
        # Ordenar por valor promedio descendente
        genre_avgs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nCluster {cluster_id} ({df_cluster.count()} usuarios):")
        print("  Top 5 géneros preferidos:")
        for genre, avg_val in genre_avgs[:5]:
            print(f"    - {genre}: {avg_val:.2f}")


def main():
    spark = build_spark_session()
    bucket_path = get_bucket_path()
    output_path = get_output_path()
    
    try:
        print(f"Spark Master: {spark.sparkContext.master}")
        print(f"Input path: {bucket_path}")
        print(f"Output path: {output_path}")
        
        # 1. Cargar datos
        df_ratings, df_movies = load_data(spark, bucket_path)
        
        # 2. Análisis exploratorio
        exploratory_analysis(df_ratings)
        
        # 3. Construir features
        df_features, genre_columns = build_features(df_ratings, df_movies)
        
        # 4. Aplicar K-Means
        df_predictions, best_k, genre_columns = apply_kmeans(spark, df_features, genre_columns, output_path)
        
        # 5. Analizar resultados
        analyze_clusters(df_predictions, best_k, df_features, genre_columns)
        
        print("\n✓ Proceso completado exitosamente")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()