import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, explode, array, lit, when, sum as spark_sum, avg, struct, array_sort, element_at, format_string, row_number, desc, round as spark_round
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


def build_features(df_ratings, df_movies):
    """
    CONSTRUCTION DE FEATURES PARA K-MEANS
    
    Entrada: ratings (userId, movieId, rating) + movies (movieId, genres)
    Salida: matriz usuario-género con weighted ratings
    
    Proceso:
    1. Filtrar ratings positivos (>3, es decir 4-5) para evitar ruido de reseñas neutras/negativas.
    2. Calcular weighted_rating = rating / num_géneros por película.
    3. Pivotear por género para crear 18 features por usuario.
    4. Rezulta en 18D feature space: [Drama_score, Comedy_score, ..., Western_score]
    """
    genre_columns = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical",
        "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
    ]

    # PASO 1: Filtrar SOLO ratings positivos (rating > 3 significa 4 o 5)
    # Justificación: Captures true preferences, elimina ruido de ratings neutrales/negativos
    df_positive = df_ratings.filter(col("rating") > 3)

    # PASO 2: Preparar mapping género-película
    # Cada película puede tener múltiples géneros; necesitamos expandir.
    genre_exprs = [when(col(g) == 1, lit(g)) for g in genre_columns]
    genre_count_col = sum(col(g) for g in genre_columns)

    df_movie_genres = df_movies.select(
        "movieId",
        genre_count_col.alias("genre_count"),  # Ej: película "Drama+Comedy" tiene genre_count=2
        explode(array(*genre_exprs)).alias("genre")  # Explode: una fila por género
    ).filter(col("genre").isNotNull())

    # PASO 3: Calcular WEIGHTED RATING por género
    # weighted_rating = rating / genre_count normaliza impacto de películas multi-género
    # Ej: Si usuario ratea 5 a "Drama+Comedy", contribuye 2.5 a Drama y 2.5 a Comedy, no 5 a cada una
    df_user_genre = df_positive.join(df_movie_genres, on="movieId") \
        .withColumn("weighted_rating", col("rating") / col("genre_count")) \
        .groupBy("userId").pivot("genre").agg(spark_sum("weighted_rating")).fillna(0)  # fillna(0): usuarios sin ratings en ciertos géneros

    print(f"\n=== FEATURES CONSTRUIDOS ===")
    print(f"Usuarios con features: {df_user_genre.count()}")
    print(f"Géneros (features): {len(genre_columns)}")
    
    return df_user_genre, genre_columns


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
    input_cols = [c for c in df_features.columns if c != "userId"]
    
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
    # - Si top_score <= 1.0: usuario con bajo consumo global -> "Perfil ligero"
    # - Si diferencia entre top2 <= 15%: usuario con preferencias distribuidas -> "Mixto"
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
            when(col("top_score") <= 1.0, lit("Perfil ligero"))  # Bajo consumo
            .when(
                (col("top_score") - col("second_score")) <= (col("top_score") * lit(0.15)),
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
            print(f"    - {genre}: {avg_val:.2f}")

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
    4. build_features() - Feature engineering: weighted ratings por género
    5. apply_kmeans() - Entrenar K-Means, seleccionar K óptimo (Silhouette Score)
    6. analyze_clusters() - Interpretación: qué caracteriza cada cluster
    7. Guardar resultados en GCS
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
        df_features, genre_columns = build_features(df_ratings, df_movies)
        
        # PASO 4: Entrenar K-Means y seleccionar K óptimo
        # Usa distancia euclidiana en espacio normalizado
        df_predictions, best_k, genre_columns = apply_kmeans(spark, df_features, genre_columns, output_path)
        
        # PASO 5: Análisis e interpretación de clusters
        # Responde: ¿qué usuarios están en cada cluster? ¿qué géneros los caracterizan?
        analyze_clusters(df_predictions, best_k, df_features, genre_columns)
        
        print("\n✓ Proceso completado exitosamente")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()