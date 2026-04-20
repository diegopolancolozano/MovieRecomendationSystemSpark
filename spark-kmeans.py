import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, count, explode, array, lit, when, sum as spark_sum, avg, struct, array_sort, element_at, format_string, row_number, desc, round as spark_round
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

"""
SEGMENTACIÓN DE USUARIOS CON K-MEANS SOBRE MOVIELENS 100K

OBJETIVO:
- Aplicar técnicas de procesamiento distribuido (Spark) para identificar grupos de usuarios similares.
- Usar K-Means con distancia euclidiana normalizada en un entorno distribuido (GCP).

ESTRATEGIA DE FEATURE ENGINEERING:
1. Se filtran SOLO ratings positivos (rating > 3, es decir 4 y 5) para capturar preferencias reales.
2. Se calcula weighted_rating = rating / cantidad_géneros_película
   - Normaliza el impacto de películas multi-género vs mono-género.
   - Ej: Drama+Comedy no pesa el doble que Drama puro.
3. Resultado: 18 features (uno por género: Drama, Comedy, Action, etc.)

MÉTRICA DE DISTANCIA:
- Euclidiana en espacio normalizado por StandardScaler (media=0, desv=1)
- Todos los géneros con igual peso: d = sqrt(Σ(genre_i_u1 - genre_i_u2)²)

SELECCIÓN DE K:
- Se prueba K=3, 5, 8 y se evalúa con Silhouette Score [-1, 1].
- K=3 elegido: mejor score (0.5751), indica clusters bien separados y cohesivos.
"""


def build_spark_session() -> SparkSession:
    master_url = os.getenv("SPARK_MASTER_URL", "spark://127.0.0.1:7077")
    
    return (
        SparkSession.builder
        .appName("MovieLens_KMeans_Clustering")
        .master(master_url)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.debug.maxToStringFields", "200")
        .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.11")
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .getOrCreate()
    )


def get_bucket_path() -> str:
    bucket = os.getenv("MOVIELENS_GCS_BUCKET")
    if bucket:
        bucket = bucket.replace("gs://", "").strip("/\\")
        return f"gs://{bucket}/ml-100k"
    return "data/ml-100k"


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


def build_movie_genre_mapping(df_movies, genre_columns):
    """
    Crea una tabla movieId-genre con normalización por cantidad de géneros.

    Cada película se expande en N filas (una por género activo) y guarda genre_count
    para repartir peso cuando una película pertenece a múltiples géneros.
    """
    genre_exprs = [when(col(g) == 1, lit(g)) for g in genre_columns]
    genre_count_col = sum(col(g) for g in genre_columns)

    return (
        df_movies
        .select(
            "movieId",
            "title",
            genre_count_col.alias("genre_count"),
            explode(array(*genre_exprs)).alias("genre")
        )
        .filter(col("genre").isNotNull())
    )


def build_features(df_ratings, df_movies):
    """
    CONSTRUCTION DE FEATURES PARA K-MEANS
    
    Entrada: ratings (userId, movieId, rating) + movies (movieId, genres)
    Salida: matriz usuario-género con weighted ratings
    
     Proceso:
     1. Expandir películas por género con normalización por multi-género.
     2. Calcular para cada usuario-género el porcentaje positivo ponderado:
         positive_rate = weighted_positive / weighted_total
     3. Aplicar suavizado bayesiano para géneros con pocos datos por usuario.
     4. Pivotear por género para crear 18 features por usuario en rango [0, 1].
    """
    genre_columns = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical",
        "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
    ]

    # Hiperparámetro de suavizado para evitar sesgo con pocos ratings.
    alpha = float(os.getenv("GENRE_SMOOTHING_ALPHA", "5"))

    # PASO 1: Preparar mapping género-película.
    df_movie_genres = build_movie_genre_mapping(df_movies, genre_columns)

    # PASO 2: Métricas usuario-género ponderadas por cantidad de géneros.
    # - weighted_positive suma 1/genre_count cuando rating > 3.
    # - weighted_total suma 1/genre_count para toda reseña.
    df_user_genre_stats = (
        df_ratings
        .join(df_movie_genres.select("movieId", "genre", "genre_count"), on="movieId")
        .withColumn("is_positive", when(col("rating") > 3, lit(1.0)).otherwise(lit(0.0)))
        .withColumn("weighted_positive", col("is_positive") / col("genre_count"))
        .withColumn("weighted_total", lit(1.0) / col("genre_count"))
        .groupBy("userId", "genre")
        .agg(
            spark_sum("weighted_positive").alias("positive_weight"),
            spark_sum("weighted_total").alias("total_weight")
        )
    )

    # PASO 3: Prior global por género para suavizado bayesiano.
    df_global_genre_rates = (
        df_user_genre_stats
        .groupBy("genre")
        .agg((spark_sum("positive_weight") / spark_sum("total_weight")).alias("global_positive_rate"))
    )

    # smoothed_rate = (positive_weight + alpha * global_positive_rate) / (total_weight + alpha)
    df_user_genre_smoothed = (
        df_user_genre_stats
        .join(df_global_genre_rates, on="genre", how="left")
        .withColumn(
            "smoothed_positive_rate",
            (col("positive_weight") + lit(alpha) * col("global_positive_rate")) /
            (col("total_weight") + lit(alpha))
        )
    )

    # PASO 4: Pivot por género para usar en clustering/recomendación.
    df_user_genre = (
        df_user_genre_smoothed
        .groupBy("userId")
        .pivot("genre")
        .agg(avg("smoothed_positive_rate"))
        .fillna(0.0)
    )

    # Garantizar las 18 columnas aunque un género sea raro en un split concreto.
    for genre in genre_columns:
        if genre not in df_user_genre.columns:
            df_user_genre = df_user_genre.withColumn(genre, lit(0.0))

    df_user_genre = df_user_genre.select("userId", *genre_columns)

    print(f"\n=== FEATURES CONSTRUIDOS ===")
    print(f"Usuarios con features: {df_user_genre.count()}")
    print(f"Géneros (features): {len(genre_columns)}")
    print(f"Suavizado bayesiano alpha: {alpha}")
    
    return df_user_genre, genre_columns, df_movie_genres


def build_movie_quality_profiles(df_ratings, df_movies, df_movie_genres, output_path):
    """
    Calcula calidad de películas a nivel reseña y ranking por género.

    Métricas por película:
    - avg_rating
    - positive_ratio (rating > 3)
    - rating_count
    - bayesian_rating para estabilizar películas con pocos votos
    - quality_score normalizado [0, 1]
    """
    beta = float(os.getenv("MOVIE_BAYES_BETA", "20"))
    top_per_genre = int(os.getenv("TOP_MOVIES_PER_GENRE", "10"))

    df_movie_stats = (
        df_ratings
        .groupBy("movieId")
        .agg(
            avg("rating").alias("avg_rating"),
            (spark_sum(when(col("rating") > 3, lit(1.0)).otherwise(lit(0.0))) / count("*")).alias("positive_ratio"),
            count("*").alias("rating_count")
        )
    )

    global_avg_rating = df_ratings.select(avg("rating").alias("global_avg_rating")).first()["global_avg_rating"]

    df_movie_scores = (
        df_movie_stats
        .withColumn(
            "bayesian_rating",
            (col("avg_rating") * col("rating_count") + lit(beta * global_avg_rating)) /
            (col("rating_count") + lit(beta))
        )
        .withColumn("quality_score", col("bayesian_rating") / lit(5.0))
        .join(df_movies.select("movieId", "title"), on="movieId", how="left")
    )

    genre_rank_window = Window.partitionBy("genre").orderBy(
        desc("quality_score"),
        desc("positive_ratio"),
        desc("rating_count")
    )

    df_top_movies_by_genre = (
        df_movie_scores
        .join(df_movie_genres.select("movieId", "genre").dropDuplicates(), on="movieId", how="inner")
        .withColumn("rank_in_genre", row_number().over(genre_rank_window))
        .filter(col("rank_in_genre") <= top_per_genre)
        .select(
            "genre",
            "rank_in_genre",
            "movieId",
            "title",
            spark_round(col("avg_rating"), 3).alias("avg_rating"),
            spark_round(col("positive_ratio"), 3).alias("positive_ratio"),
            "rating_count",
            spark_round(col("bayesian_rating"), 3).alias("bayesian_rating")
        )
        .orderBy("genre", "rank_in_genre")
    )

    df_movie_scores.write.mode("overwrite").csv(f"{output_path}/movie_quality_scores")
    df_top_movies_by_genre.write.mode("overwrite").csv(f"{output_path}/top_movies_by_genre")

    print("\n=== CALIDAD DE PELÍCULAS (NIVEL RESEÑAS) ===")
    print(f"Promedio global rating: {global_avg_rating:.3f}")
    print(f"Bayesian beta: {beta}")
    print(f"Top por género guardado en: {output_path}/top_movies_by_genre")

    return df_movie_scores, df_top_movies_by_genre


def build_user_recommendations(df_ratings, df_user_genre, genre_columns, df_movie_genres, df_movie_scores, output_path):
    """
    Genera recomendaciones personalizadas Top-N por usuario.

    Estrategia híbrida:
    final_score = w_pref * affinity_score + w_quality * quality_score + w_positive * positive_ratio

    - affinity_score: afinidad usuario-película derivada de preferencias por género.
    - quality_score: calidad global estabilizada de película (bayesiana).
    - positive_ratio: proporción global de reseñas positivas de la película.
    """
    top_n = int(os.getenv("TOP_N_RECOMMENDATIONS", "10"))
    w_pref = float(os.getenv("WEIGHT_PREFERENCE", "0.6"))
    w_quality = float(os.getenv("WEIGHT_QUALITY", "0.3"))
    w_positive = float(os.getenv("WEIGHT_POSITIVE", "0.1"))

    # Normalizar pesos para robustez cuando se cambian por entorno.
    weight_sum = w_pref + w_quality + w_positive
    if weight_sum == 0:
        w_pref, w_quality, w_positive = 0.6, 0.3, 0.1
    else:
        w_pref, w_quality, w_positive = w_pref / weight_sum, w_quality / weight_sum, w_positive / weight_sum

    # Unpivot de matriz usuario-género -> formato largo (userId, genre, genre_preference).
    preference_structs = [struct(lit(g).alias("genre"), col(g).alias("genre_preference")) for g in genre_columns]
    df_user_genre_long = (
        df_user_genre
        .select("userId", explode(array(*preference_structs)).alias("pref"))
        .select("userId", col("pref.genre").alias("genre"), col("pref.genre_preference").alias("genre_preference"))
    )

    # Afinidad usuario-película como promedio de preferencias de sus géneros.
    df_user_movie_affinity = (
        df_user_genre_long
        .join(df_movie_genres.select("movieId", "genre").dropDuplicates(), on="genre", how="inner")
        .groupBy("userId", "movieId")
        .agg(avg("genre_preference").alias("affinity_score"))
    )

    # Excluir películas ya vistas por cada usuario.
    df_seen_movies = df_ratings.select("userId", "movieId").distinct()

    df_candidates = (
        df_user_movie_affinity
        .join(
            df_movie_scores.select("movieId", "title", "quality_score", "positive_ratio", "rating_count"),
            on="movieId",
            how="inner"
        )
        .join(df_seen_movies, on=["userId", "movieId"], how="left_anti")
        .withColumn(
            "final_score",
            lit(w_pref) * col("affinity_score") +
            lit(w_quality) * col("quality_score") +
            lit(w_positive) * col("positive_ratio")
        )
    )

    rank_window = Window.partitionBy("userId").orderBy(
        desc("final_score"),
        desc("quality_score"),
        desc("rating_count")
    )

    df_top_recommendations = (
        df_candidates
        .withColumn("rank", row_number().over(rank_window))
        .filter(col("rank") <= top_n)
        .select(
            "userId",
            "rank",
            "movieId",
            "title",
            spark_round(col("final_score"), 4).alias("final_score"),
            spark_round(col("affinity_score"), 4).alias("affinity_score"),
            spark_round(col("quality_score"), 4).alias("quality_score"),
            spark_round(col("positive_ratio"), 4).alias("positive_ratio"),
            "rating_count"
        )
        .orderBy("userId", "rank")
    )

    df_top_recommendations.write.mode("overwrite").csv(f"{output_path}/recommendations_top{top_n}")

    print("\n=== RECOMENDACIONES PERSONALIZADAS ===")
    print(f"Pesos usados: preference={w_pref:.2f}, quality={w_quality:.2f}, positive={w_positive:.2f}")
    print(f"Top-N por usuario: {top_n}")
    print(f"Recomendaciones guardadas en: {output_path}/recommendations_top{top_n}")

    return df_top_recommendations


def apply_kmeans(spark, df_features, genre_columns, output_path):
    """
    K-MEANS CLUSTERING CON NORMALIZACIÓN
    
    Proceso:
    1. Ensamblar 18 features en un vector denso (VectorAssembler)
    2. Normalizar con StandardScaler: media=0, desv=1 para cada feature
       -> Garantiza que todos los géneros tengan igual peso en distancia euclidiana
    3. Entrenar K-Means para K=[3,5,8] con seed=42 (reproducible)
    4. Evaluar con Silhouette Score: mide cohesión vs separación
       -> Rango [-1, 1]; >0.5 indica clusters bien definidos
    5. Seleccionar K con mejor Silhouette Score
    """
    input_cols = [c for c in genre_columns if c in df_features.columns]
    
    # PASO 1: VectorAssembler - Convertir 18 columnas en UN vector denso
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features")
    df_assembled = assembler.transform(df_features)
    
    # PASO 2: StandardScaler - Normalizar a media=0, desv=1
    # Si no se pone esto, géneros con valores más altos dominarían la métrica
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    print("\n=== APLICANDO K-MEANS ===")
    
    # PASO 3: Entrenar K-Means para múltiples K y seleccionar mejor
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    
    best_k = None
    best_score = -1
    best_predictions = None
    
    # Probar K=3, 5, 8 y elegir el mejor por Silhouette Score
    for k in [3, 5, 8]:
        kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=20)
        model = kmeans.fit(df_scaled)  # Distancia euclidiana utilizada internamente
        predictions = model.transform(df_scaled)
        
        # Silhouette Score: [-1, 1]. Valores > 0.5 indican clusters bien separados
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
    """
    INTERPRETACIÓN DE CLUSTERS
    
    Objetivo: Caracterizar cada cluster por:
    1. Tamaño (número de usuarios)
    2. Géneros preferidos (weighted ratings promedio)
    3. Tipos de usuarios (intensidad de consumo)
    
    Salida: Tablas que muestran qué caracteriza a cada grupo.
    """
    print(f"\n=== ANALISIS DE CLUSTERS (K={k}) ===")
    
    # PASO 1: Distribución de usuarios
    cluster_dist = df_predictions.groupBy("prediction").count().orderBy("prediction")
    print("\nDistribución de usuarios por cluster:")
    cluster_dist.show()
    
    # PASO 2: Combinar asignación de cluster con features originales para análisis detallado
    df_analysis = df_predictions.select("userId", "prediction").join(df_features, on="userId")

    # PASO 3: Perfilar cada usuario por top 2 géneros y categoría de preferencia
    # Lógica:
    # - Si top_score <= 45%: usuario con preferencia débil -> "Perfil ligero"
    # - Si diferencia entre top2 <= 10 puntos: usuario con preferencias distribuidas -> "Mixto"
    # - Sino: usuario con género dominante -> "Fan de [género]"
    genre_scores = [struct(col(g).alias("score"), lit(g).alias("genre")) for g in genre_columns]
    df_user_profiles = (
        df_analysis
        .withColumn("sorted_genres", array_sort(array(*genre_scores)))  # Ordenar géneros by score
        .withColumn("top_genre", element_at(col("sorted_genres"), -1).getField("genre"))  # Último = máximo
        .withColumn("top_score", element_at(col("sorted_genres"), -1).getField("score"))
        .withColumn("second_genre", element_at(col("sorted_genres"), -2).getField("genre"))  # Penúltimo = 2do
        .withColumn("second_score", element_at(col("sorted_genres"), -2).getField("score"))
        .withColumn(
            "user_type",
            when(col("top_score") <= 0.45, lit("Perfil ligero"))  # Afinidad baja (escala 0-1)
            .when(
                (col("top_score") - col("second_score")) <= lit(0.10),
                format_string("Mixto: %s y %s", col("top_genre"), col("second_genre"))  # Preferencias balanceadas
            )
            .otherwise(format_string("Fan de %s", col("top_genre")))  # Género dominante
        )
    )

    # PASO 4: Calcular weighted_rating promedio por género en cada cluster
    # Esto revela qué géneros definen cada grupo
    print("\nCaracterísticas promedio por cluster (top géneros):")
    cluster_profiles = []
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
            print(f"    - {genre}: {avg_val:.2%}")

        # Crear descripción compacta de cada cluster (top 3 géneros)
        top3 = [genre for genre, _ in genre_avgs[:3]]
        cluster_profile = f"Predomina {top3[0]} | Secundarios: {top3[1]}, {top3[2]}"
        cluster_profiles.append((cluster_id, cluster_profile))
        print(f"  Caracterización: {cluster_profile}")

    # PASO 5: Crear tabla resumen cluster y uniría a perfiles de usuario
    spark = df_predictions.sparkSession
    df_cluster_profiles = spark.createDataFrame(cluster_profiles, ["prediction", "cluster_profile"])
    df_user_profiles = df_user_profiles.join(df_cluster_profiles, on="prediction", how="left")

    # PASO 6: Mostrar muestra de usuarios - ayuda a entender concretamente quién está en cada cluster
    print("\nMuestra de usuarios por cluster (5 por cluster):")
    sample_window = Window.partitionBy("prediction").orderBy("userId")
    df_user_profiles.withColumn("sample_rank", row_number().over(sample_window)) \
        .filter(col("sample_rank") <= 5) \
        .select("prediction", "userId", "user_type", "top_genre", "second_genre") \
        .orderBy("prediction", "userId") \
        .show(truncate=False)

    # PASO 7: Tabla final compacta por cluster
    print("\nResumen limpio por cluster:")
    df_cluster_sizes = df_user_profiles.groupBy("prediction", "cluster_profile").count() \
        .withColumnRenamed("count", "total_users")

    df_type_counts = df_user_profiles.groupBy("prediction", "user_type").count()
    dominant_window = Window.partitionBy("prediction").orderBy(desc("count"), col("user_type"))
    df_dominant_type = df_type_counts.withColumn("rank", row_number().over(dominant_window)) \
        .filter(col("rank") == 1) \
        .select(
            "prediction",
            col("user_type").alias("tipo_dominante"),
            col("count").alias("dominant_users")
        )

    df_cluster_sizes.join(df_dominant_type, on="prediction", how="left") \
        .withColumn("pct_tipo_dominante", spark_round(col("dominant_users") * lit(100.0) / col("total_users"), 1)) \
        .select("prediction", "total_users", "cluster_profile", "tipo_dominante", "pct_tipo_dominante") \
        .orderBy("prediction") \
        .show(truncate=False)


def main():
    """
    PIPELINE PRINCIPAL: K-MEANS CLUSTERING EN SPARK
    
    Flujo:
    1. build_spark_session() - Conectar a cluster Spark y configurar GCS
    2. load_data() - Cargar MovieLens ratings y movies desde GCS
    3. exploratory_analysis() - EDA: estadísticas básicas de dataset
    4. build_features() - Feature engineering: % positivos por género (suavizado)
    5. apply_kmeans() - Entrenar K-Means, seleccionar K óptimo (Silhouette Score)
    6. analyze_clusters() - Interpretación: qué caracteriza cada cluster
    7. build_movie_quality_profiles() - Calidad película por reseñas + top por género
    8. build_user_recommendations() - Top-N personalizado por usuario
    9. Guardar resultados en GCS
    """
    spark = build_spark_session()
    log_level = os.getenv("SPARK_LOG_LEVEL", "WARN").upper()
    spark.sparkContext.setLogLevel(log_level)
    bucket_path = get_bucket_path()
    output_path = get_output_path()
    
    try:
        print(f"Spark Master: {spark.sparkContext.master}")
        print(f"Spark log level: {log_level}")
        print(f"Input path: {bucket_path}")
        print(f"Output path: {output_path}")
        
        # PASO 1: Cargar datos del dataset MovieLens 100K
        df_ratings, df_movies = load_data(spark, bucket_path)
        
        # PASO 2: Análisis exploratorio - entender dataset
        exploratory_analysis(df_ratings)
        
        # PASO 3: Construcción de features - preparar datos para clustering
        # Convierte ratings + géneros en matriz usuario-género normalizada
        df_features, genre_columns, df_movie_genres = build_features(df_ratings, df_movies)
        
        # PASO 4: Entrenar K-Means y seleccionar K óptimo
        # Usa distancia euclidiana en espacio normalizado
        df_predictions, best_k, genre_columns = apply_kmeans(spark, df_features, genre_columns, output_path)
        
        # PASO 5: Análisis e interpretación de clusters
        # Responde: ¿qué usuarios están en cada cluster? ¿qué géneros los caracterizan?
        analyze_clusters(df_predictions, best_k, df_features, genre_columns)

        # PASO 6: Calidad de películas a nivel reseña y ranking por género
        df_movie_scores, _ = build_movie_quality_profiles(df_ratings, df_movies, df_movie_genres, output_path)

        # PASO 7: Recomendaciones personalizadas Top-N por usuario
        build_user_recommendations(
            df_ratings,
            df_features,
            genre_columns,
            df_movie_genres,
            df_movie_scores,
            output_path
        )
        
        print("\n✓ Proceso completado exitosamente")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()