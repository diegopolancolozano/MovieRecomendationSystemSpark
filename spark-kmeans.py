"""
Sistema de Recomendación Basado en Clusters de Usuarios — Versión GCP / Cluster Spark
=======================================================================================
Dataset  : MovieLens 1M (ratings.dat, movies.dat)
Objetivo : Agrupar usuarios con K-Means según sus preferencias de género y
           recomendar películas no vistas usando la estrategia Confidence Hybrid.

Objetivos del laboratorio cubiertos:
  1. División Train/Test (80/20) con seed reproducible.
  2. Construcción de clusters de usuarios con K-Means sobre features de TRAIN.
  3. Generación de Top-N recomendaciones por usuario usando su cluster.
  4. Exclusión de películas ya vistas en TRAIN (left_anti join).
  5. Evaluación con Precision@N y Recall@N sobre TEST para múltiples valores de K.
  6. Guardado de recomendaciones y métricas en archivos JSON.

Despliegue:
  - En GCP/Dataproc: configurar SPARK_MASTER_URL y MOVIELENS_GCS_BUCKET como variables de entorno.
  - Localmente: usar spark-kmeans-local.py (sin GCS, con corrección de versión Python en Windows).
"""

import argparse
import csv
import json
import math
import os
from functools import reduce
from typing import Dict, List, Tuple

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, LongType, StructField, StructType


def parse_args() -> argparse.Namespace:
    """
    Parsea argumentos de línea de comandos y variables de entorno.
    Las variables de entorno sirven como fallback cuando no se pasan flags CLI,
    lo que facilita el despliegue en GCP sin modificar el script.
    """
    parser = argparse.ArgumentParser(description="Cluster based recommendation with Spark")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR"),
        help="Directory containing ratings.dat and movies.dat.",
    )
    parser.add_argument(
        "--output-path",
        default=os.getenv("OUTPUT_PATH"),
        help="Output path for generated files.",
    )
    parser.add_argument(
        "--k-values",
        default=os.getenv("K_VALUES") or "3,5,8,10",
        help="Comma-separated K values for KMeans, e.g. 3,5,8,10",
    )
    parser.add_argument(
        "--strategy",
        default=os.getenv("RECOMMENDATION_STRATEGY") or "confidence_hybrid",
        help="Recommendation strategy: affinity or confidence_hybrid",
    )
    parser.add_argument(
        "--demo-if-missing",
        action="store_true",
        help="Run with small demo data if MovieLens files are missing.",
    )
    return parser.parse_args()


def env_float(name: str, default: float) -> float:
    """Lee una variable de entorno como float; si no existe o es inválida, usa el valor por defecto."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[WARN] Invalid float for {name}={value}. Using default={default}")
        return default


def env_int(name: str, default: int) -> int:
    """Lee una variable de entorno como int; si no existe o es inválida, usa el valor por defecto."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARN] Invalid int for {name}={value}. Using default={default}")
        return default


def parse_k_values(raw: str) -> List[int]:
    """Convierte una cadena CSV de K values (ej. '3,5,8,10') a lista de enteros válidos (>= 2)."""
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            k = int(token)
            if k >= 2:
                values.append(k)
        except ValueError:
            print(f"[WARN] Ignoring invalid K value: {token}")
    values = sorted(list(set(values)))
    return values if values else [3, 5]


def parse_strategy(raw: str) -> str:
    """Valida la estrategia de recomendación; si no es válida usa 'confidence_hybrid' por defecto."""
    strategy = (raw or "").strip().lower()
    valid = {"affinity", "confidence_hybrid"}
    if strategy in valid:
        return strategy

    if strategy:
        print(f"[WARN] Invalid strategy '{raw}'. Using confidence_hybrid")
    return "confidence_hybrid"


def get_input_path(cli_data_dir: str) -> str:
    """
    Resuelve la ruta de datos con esta prioridad:
      1. Argumento CLI --data-dir
      2. Variable de entorno MOVIELENS_GCS_BUCKET → ruta gs://bucket/ml-1m
      3. Directorio local 'ml-1m' (fallback)
    """
    if cli_data_dir:
        return cli_data_dir

    bucket = os.getenv("MOVIELENS_GCS_BUCKET")
    if bucket:
        bucket = bucket.replace("gs://", "").strip("/\\")
        return f"gs://{bucket}/ml-1m"

    return "ml-1m"


def get_output_path(cli_output_path: str) -> str:
    """
    Resuelve la ruta de salida con la misma prioridad que get_input_path:
      CLI → GCS bucket → directorio local 'output'.
    """
    if cli_output_path:
        return cli_output_path

    bucket = os.getenv("MOVIELENS_GCS_BUCKET")
    if bucket:
        bucket = bucket.replace("gs://", "").strip("/\\")
        return f"gs://{bucket}/output"

    return "output"


def build_spark_session() -> SparkSession:
    """
    Crea y configura la SparkSession.
    - En GCP/Dataproc: SPARK_MASTER_URL apunta al master del cluster.
    - En local: usa local[*] por defecto.
    - El conector GCS solo se activa si MOVIELENS_GCS_BUCKET está definido,
      evitando descargas innecesarias en ejecuciones locales.
    """
    master_url = os.getenv("SPARK_MASTER_URL", "local[*]")

    builder = (
        SparkSession.builder.appName("MovieLens_Recommendation_Clusters")
        .master(master_url)
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.default.parallelism", os.getenv("SPARK_DEFAULT_PARALLELISM", "8"))
        .config("spark.sql.debug.maxToStringFields", "200")
    )

    if os.getenv("MOVIELENS_GCS_BUCKET"):
        builder = (
            builder.config(
                "spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.11"
            )
            .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
            .config(
                "spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
            )
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN").upper())
    return spark


def has_local_movielens_files(data_path: str) -> bool:
    """Verifica si los archivos de MovieLens existen en la ruta indicada.
    Las rutas GCS siempre devuelven True (la verificación la hace Spark al leer)."""
    if data_path.startswith("gs://"):
        return True

    ratings_path = os.path.join(data_path, "ratings.dat")
    movies_path = os.path.join(data_path, "movies.dat")
    return os.path.exists(ratings_path) and os.path.exists(movies_path)


def load_movielens_data(spark: SparkSession, data_path: str) -> Tuple[DataFrame, DataFrame]:
    """
    Carga ratings.dat y movies.dat usando el SparkContext (RDD API).
    El separador '::' no es soportado como delimitador de un solo carácter en
    el CSV reader de Spark, por lo que se parsea manualmente con split("::").
    """
    ratings_schema = StructType(
        [
            StructField("userId", IntegerType(), False),
            StructField("movieId", IntegerType(), False),
            StructField("rating", FloatType(), False),
            StructField("timestamp", LongType(), False),
        ]
    )

    ratings_raw = spark.sparkContext.textFile(f"{data_path}/ratings.dat")
    ratings = (
        ratings_raw.map(lambda line: line.split("::"))
        .map(lambda parts: (int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])))
        .toDF(ratings_schema)
    )

    movies_raw = spark.sparkContext.textFile(f"{data_path}/movies.dat")
    movies = (
        movies_raw.map(lambda line: line.split("::"))
        .map(lambda parts: (int(parts[0]), parts[1], parts[2]))
        .toDF(["movieId", "title", "genres"])
    )

    return ratings, movies


def load_demo_data(spark: SparkSession) -> Tuple[DataFrame, DataFrame]:
    """
    Genera un dataset de demostración mínimo (6 usuarios, 8 películas, 24 ratings).
    Se usa cuando --demo-if-missing está activo y no se encuentran los archivos reales.
    Útil para verificar que el pipeline corre sin necesidad del dataset completo.
    """
    ratings_rows = [
        (1, 1, 5.0, 978300001),
        (1, 2, 4.0, 978300010),
        (1, 3, 2.0, 978300020),
        (1, 6, 4.5, 978300030),
        (2, 1, 4.0, 978301001),
        (2, 4, 5.0, 978301010),
        (2, 5, 3.0, 978301020),
        (2, 7, 4.5, 978301030),
        (3, 2, 2.0, 978302001),
        (3, 3, 4.5, 978302010),
        (3, 4, 4.0, 978302020),
        (3, 8, 3.5, 978302030),
        (4, 5, 4.5, 978303001),
        (4, 6, 2.0, 978303010),
        (4, 7, 4.0, 978303020),
        (4, 8, 5.0, 978303030),
        (5, 1, 3.5, 978304001),
        (5, 3, 4.0, 978304010),
        (5, 6, 5.0, 978304020),
        (5, 7, 2.5, 978304030),
        (6, 2, 4.5, 978305001),
        (6, 4, 3.5, 978305010),
        (6, 5, 4.0, 978305020),
        (6, 8, 2.0, 978305030),
    ]

    movies_rows = [
        (1, "Action One", "Action|Thriller"),
        (2, "Comedy One", "Comedy|Romance"),
        (3, "Drama One", "Drama"),
        (4, "SciFi One", "Sci-Fi|Action"),
        (5, "Family One", "Animation|Children's"),
        (6, "Doc One", "Documentary|Drama"),
        (7, "Thrill One", "Thriller|Drama"),
        (8, "Love One", "Romance|Comedy"),
    ]

    ratings = spark.createDataFrame(
        ratings_rows, ["userId", "movieId", "rating", "timestamp"]
    )
    movies = spark.createDataFrame(movies_rows, ["movieId", "title", "genres"])
    return ratings, movies


def exploratory_analysis(ratings: DataFrame, movies: DataFrame) -> None:
    """
    Imprime estadísticas básicas del dataset para verificar la carga correcta de datos:
    número de usuarios únicos, películas, total de ratings y distribución de calificaciones.
    """
    users = ratings.select("userId").distinct().count()
    movie_count = movies.count()
    ratings_count = ratings.count()

    print("\n=== EDA ===")
    print(f"Users      : {users}")
    print(f"Movies     : {movie_count}")
    print(f"Ratings    : {ratings_count}")
    print("Ratings distribution:")
    ratings.groupBy("rating").count().orderBy("rating").show(truncate=False)


def safe_genre_column(genre_name: str) -> str:
    """
    Convierte un nombre de género a un nombre de columna Spark válido.
    Elimina caracteres especiales que causan errores de parsing.
    Ejemplo: "Film-Noir" → "genre_film_noir", "Children's" → "genre_childrens"
    """
    cleaned = genre_name.lower()
    for old, new in [("-", "_"), ("'", ""), (" ", "_"), ("(", ""), (")", "")]:
        cleaned = cleaned.replace(old, new)
    return f"genre_{cleaned}"


def build_user_genre_features(
    train_ratings: DataFrame,
    movies: DataFrame,
    positive_threshold: float,
) -> Tuple[DataFrame, DataFrame, List[str]]:
    """
    Construye dos matrices de features a partir de los datos de TRAIN:
      - user_genre_matrix : userId × género → suma de weighted_rating
        weighted_rating = rating / número_de_géneros (evita sesgo por películas multi-género)
        Solo ratings positivos (> positive_threshold) capturan preferencias reales.
      - movie_genre_matrix: movieId × género → presencia binaria (1.0 si pertenece al género)

    Retorna (user_genre_matrix, movie_genre_matrix, feature_cols ordenadas).
    """
    movies_expanded = (
        movies.withColumn("genre_array", F.split(F.col("genres"), "\\|"))
        .withColumn("genre_count", F.size(F.col("genre_array")))
        .select("movieId", "genre_count", F.explode(F.col("genre_array")).alias("genre"))
    )

    positive = train_ratings.filter(F.col("rating") > F.lit(positive_threshold))
    train_genre = (
        positive.join(movies_expanded, "movieId", "inner")
        .withColumn("weighted_rating", F.col("rating") / F.col("genre_count"))
    )

    genre_values = [
        row["genre"]
        for row in movies_expanded.select("genre").distinct().where(F.col("genre").isNotNull()).collect()
    ]

    user_genre_matrix = (
        train_genre.groupBy("userId")
        .pivot("genre", genre_values)
        .agg(F.sum("weighted_rating"))
        .fillna(0.0)
    )

    movie_genre_matrix = (
        movies_expanded.groupBy("movieId")
        .pivot("genre", genre_values)
        .agg(F.count("genre").cast("double"))
        .fillna(0.0)
    )

    feature_cols = []
    for genre in genre_values:
        safe = safe_genre_column(genre)
        if genre in user_genre_matrix.columns:
            user_genre_matrix = user_genre_matrix.withColumnRenamed(genre, safe)
        else:
            user_genre_matrix = user_genre_matrix.withColumn(safe, F.lit(0.0))

        if genre in movie_genre_matrix.columns:
            movie_genre_matrix = movie_genre_matrix.withColumnRenamed(genre, safe)
        else:
            movie_genre_matrix = movie_genre_matrix.withColumn(safe, F.lit(0.0))

        feature_cols.append(safe)

    return user_genre_matrix, movie_genre_matrix, sorted(feature_cols)


def run_kmeans_for_k(user_genre_matrix: DataFrame, k_value: int) -> Tuple[DataFrame, float]:
    """
    Aplica K-Means sobre los features de usuario para un valor de K dado.

    Pasos:
      1. VectorAssembler: combina columnas de género en un vector denso.
      2. StandardScaler: normaliza (media=0, std=1) para que géneros con alta
         varianza no dominen la distancia euclidiana del clustering.
      3. KMeans: agrupa usuarios en k clusters.
      4. ClusteringEvaluator: calcula Silhouette Score (cercano a 1 = clusters compactos).

    Retorna (DataFrame userId → cluster, silhouette score).
    """
    feature_cols = [c for c in user_genre_matrix.columns if c != "userId"]
    if not feature_cols:
        raise ValueError("No feature columns for KMeans")

    user_count = user_genre_matrix.count()
    if k_value < 2 or k_value >= user_count:
        raise ValueError(f"Invalid K={k_value} for user_count={user_count}")

    assembled = VectorAssembler(inputCols=feature_cols, outputCol="features_raw").transform(
        user_genre_matrix
    )
    scaled = StandardScaler(
        inputCol="features_raw", outputCol="features", withStd=True, withMean=True
    ).fit(assembled).transform(assembled)

    model = KMeans(
        k=k_value,
        featuresCol="features",
        predictionCol="prediction",
        seed=42,
        maxIter=40,
    ).fit(scaled)

    predictions = model.transform(scaled)
    silhouette = ClusteringEvaluator(
        predictionCol="prediction", featuresCol="features"
    ).evaluate(predictions)

    clusters = predictions.select("userId", F.col("prediction").alias("cluster"))
    return clusters, float(silhouette)


def log_cluster_distribution(clusters: DataFrame, k_value: int) -> None:
    """Imprime el tamaño de cada cluster para revisar el balance del K-Means."""
    distribution = clusters.groupBy("cluster").count().orderBy("cluster").collect()
    if not distribution:
        print(f"[WARN] K={k_value}: no cluster assignments found")
        return

    counts = [int(row["count"]) for row in distribution]
    total = sum(counts)
    print(f"[INFO] K={k_value} cluster distribution:")
    for row in distribution:
        cluster_id = int(row["cluster"])
        num_users = int(row["count"])
        pct = 100.0 * num_users / total if total else 0.0
        print(f"[INFO]   cluster={cluster_id} users={num_users} ({pct:.2f}%)")
    print(
        f"[INFO] K={k_value} cluster sizes -> min={min(counts)}, max={max(counts)}, avg={total / len(counts):.2f}"
    )


def generate_cluster_recommendations(
    train_ratings: DataFrame,
    clusters: DataFrame,
    user_genre_matrix: DataFrame,
    movie_genre_matrix: DataFrame,
    feature_cols: List[str],
    movies: DataFrame,
    top_n: int,
    min_cluster_votes: int,
    min_cluster_avg_rating: float,
    strategy: str,
    global_mean_rating: float,
    shrinkage_beta: float,
    weight_affinity: float,
    weight_bayesian: float,
    weight_popularity: float,
) -> DataFrame:
    """
    Genera recomendaciones Top-N para cada usuario basadas en su cluster.

    Flujo:
      1. Calcular estadísticas (avg_rating, votos) por (cluster, película) en TRAIN.
      2. Filtrar películas con suficientes votos y rating aceptable en el cluster.
      3. Excluir películas ya vistas en TRAIN con left_anti join.
      4. Calcular Affinity Score: producto punto entre preferencias de género
         del usuario y el vector binario de géneros de la película.
      5. Si strategy == 'confidence_hybrid':
           - Normalizar afinidad a [0,1] por usuario (min-max scaling).
           - Bayesian Rating: suaviza el avg del cluster con la media global
             para evitar sobreestimación de películas con pocos votos.
             Fórmula: (avg*n + global_mean*beta) / (n + beta)
           - Popularity Norm: log(votos) normalizado al máximo del dataset.
           - Score final = 0.45*affinity_norm + 0.45*bayesian/5 + 0.10*popularity_norm
      6. Rankear con Window function y devolver Top-N por usuario.
    """
    train_clustered = train_ratings.join(clusters, "userId", "inner")

    cluster_movie_scores = train_clustered.groupBy("cluster", "movieId").agg(
        F.avg("rating").alias("cluster_avg_rating"),
        F.count("*").alias("cluster_num_ratings"),
    )

    # Filtro de calidad: evitar recomendar películas con poca evidencia o mal valoradas
    cluster_movie_scores = cluster_movie_scores.filter(
        (F.col("cluster_num_ratings") >= F.lit(min_cluster_votes))
        & (F.col("cluster_avg_rating") >= F.lit(min_cluster_avg_rating))
    )

    # Películas ya vistas por cada usuario en TRAIN (a excluir de las recomendaciones)
    seen = train_ratings.select("userId", "movieId").distinct()

    # Candidatos: películas del cluster que el usuario NO ha visto en TRAIN
    candidates = clusters.join(cluster_movie_scores, "cluster", "inner")
    candidates = candidates.join(seen, ["userId", "movieId"], "left_anti")

    if feature_cols:
        # Renombrar columnas de género para evitar ambigüedad en los joins
        user_features = user_genre_matrix.select(
            "userId", *[F.col(c).alias(f"u_{c}") for c in feature_cols]
        )
        movie_features = movie_genre_matrix.select(
            "movieId", *[F.col(c).alias(f"m_{c}") for c in feature_cols]
        )

        # Producto punto: mide qué tan bien encaja la película con las preferencias del usuario
        affinity_expr = reduce(
            lambda a, b: a + b,
            [F.col(f"u_{c}") * F.col(f"m_{c}") for c in feature_cols],
        )

        candidates = (
            candidates.join(user_features, "userId", "inner")
            .join(movie_features, "movieId", "inner")
            .withColumn("affinity_score", affinity_expr)
        )
    else:
        # Sin features de género, usar el avg del cluster como proxy de afinidad
        candidates = candidates.withColumn("affinity_score", F.col("cluster_avg_rating"))

    # Score de ranking base (se sobrescribe con confidence_hybrid si aplica)
    candidates = candidates.withColumn("ranking_score", F.col("affinity_score"))

    if strategy == "confidence_hybrid":
        # Normalización min-max de afinidad POR USUARIO para comparación justa entre usuarios
        user_affinity_bounds = candidates.groupBy("userId").agg(
            F.min("affinity_score").alias("affinity_min"),
            F.max("affinity_score").alias("affinity_max"),
        )

        # Máximo de votos global: denominador para escalar la popularidad a [0,1]
        max_cluster_votes = cluster_movie_scores.agg(
            F.max("cluster_num_ratings").alias("max_votes")
        ).first()["max_votes"]
        max_cluster_votes = float(max_cluster_votes or 1.0)

        candidates = candidates.join(user_affinity_bounds, "userId", "left")

        # Normalizar afinidad a [0,1]; si min==max (todos iguales) asignar 0.5
        candidates = candidates.withColumn(
            "affinity_norm",
            F.when(
                F.col("affinity_max") > F.col("affinity_min"),
                (F.col("affinity_score") - F.col("affinity_min"))
                / (F.col("affinity_max") - F.col("affinity_min")),
            ).otherwise(F.lit(0.5)),
        )

        # Bayesian Average: (avg_cluster*n + global_mean*beta) / (n + beta)
        # Con pocos votos (n << beta) el rating se acerca a la media global (conservador).
        # Con muchos votos (n >> beta) el rating se acerca al promedio del cluster.
        candidates = candidates.withColumn(
            "bayesian_rating",
            (
                F.col("cluster_avg_rating") * F.col("cluster_num_ratings")
                + F.lit(global_mean_rating) * F.lit(shrinkage_beta)
            )
            / (F.col("cluster_num_ratings") + F.lit(shrinkage_beta)),
        )

        # Popularidad en escala logarítmica para reducir el impacto de diferencias extremas
        pop_denominator = float(math.log1p(max_cluster_votes))
        if pop_denominator <= 0.0:
            pop_denominator = 1.0

        candidates = candidates.withColumn(
            "popularity_norm",
            F.log1p(F.col("cluster_num_ratings")) / F.lit(pop_denominator),
        )

        # Score final: combinación ponderada de los tres componentes normalizados
        candidates = candidates.withColumn(
            "ranking_score",
            F.lit(weight_affinity) * F.col("affinity_norm")
            + F.lit(weight_bayesian) * (F.col("bayesian_rating") / F.lit(5.0))
            + F.lit(weight_popularity) * F.col("popularity_norm"),
        )

    rank_window = Window.partitionBy("userId").orderBy(
        F.desc("ranking_score"),
        F.desc("affinity_score"),
        F.desc("cluster_avg_rating"),
        F.desc("cluster_num_ratings"),
        F.asc("movieId"),
    )

    recs = candidates.withColumn("rank", F.row_number().over(rank_window)).filter(
        F.col("rank") <= top_n
    )

    recs = recs.join(movies.select("movieId", "title", "genres"), "movieId", "left")

    return recs.select(
        "userId",
        "cluster",
        "movieId",
        "title",
        "genres",
        "ranking_score",
        "affinity_score",
        "cluster_avg_rating",
        "cluster_num_ratings",
        "rank",
    )


def evaluate_recommendations(
    recommendations: DataFrame, test_ratings: DataFrame, relevance_threshold: float
) -> Dict[str, float]:
    """
    Calcula Precision@N y Recall@N sobre el conjunto de TEST.

    Definiciones:
      - Relevante : ítem calificado en TEST con rating >= relevance_threshold.
      - Hit       : película recomendada que el usuario calificó como relevante en TEST.
      - Precision@N = hits / N   (fracción de las N recomendaciones que son relevantes)
      - Recall@N    = hits / total_relevantes  (fracción de los relevantes recuperados)

    Inner join en la evaluación: solo incluye usuarios con al menos un ítem relevante
    en TEST, evitando que usuarios sin relevantes arrastren el promedio hacia cero.
    """
    relevant = (
        test_ratings.filter(F.col("rating") >= F.lit(relevance_threshold))
        .select("userId", "movieId")
        .withColumn("is_relevant", F.lit(1.0))
    )

    recs_top = recommendations.select("userId", "movieId")

    user_hits = (
        recs_top.join(relevant, ["userId", "movieId"], "left")
        .fillna(0.0, subset=["is_relevant"])
        .groupBy("userId")
        .agg(
            F.sum("is_relevant").alias("num_hits"),
            F.count("*").alias("num_recs"),
        )
    )

    rel_per_user = relevant.groupBy("userId").agg(F.count("*").alias("num_relevant"))

    eval_df = user_hits.join(rel_per_user, "userId", "inner")
    eval_df = eval_df.withColumn("precision", F.col("num_hits") / F.col("num_recs")).withColumn(
        "recall", F.col("num_hits") / F.col("num_relevant")
    )

    agg = eval_df.agg(
        F.avg("precision").alias("avg_precision"),
        F.avg("recall").alias("avg_recall"),
        F.count("*").alias("num_users_evaluated"),
    ).first()

    precision = float(agg["avg_precision"] or 0.0)
    recall = float(agg["avg_recall"] or 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "num_users_evaluated": int(agg["num_users_evaluated"]),
    }


def _write_csv_fallback(df: DataFrame, output_dir: str) -> None:
    """
    Escribe un DataFrame como CSV usando Python nativo.
    Se usa como fallback cuando df.write.parquet() falla por falta de
    winutils.exe (HADOOP_HOME) en Windows.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")
    columns = df.columns

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in df.toLocalIterator():
            writer.writerow([row[c] for c in columns])

    print(f"[WARN] Wrote CSV fallback: {csv_path}")


def write_dataset(df: DataFrame, output_dir: str) -> None:
    """
    Guarda un DataFrame en formato Parquet (eficiente para GCP/HDFS).
    En Windows local, si falla por HADOOP_HOME, hace fallback a CSV.
    """
    try:
        df.write.mode("overwrite").parquet(output_dir)
        print(f"[INFO] Wrote parquet: {output_dir}")
    except Exception as exc:
        msg = str(exc).lower()
        is_windows_local = (os.name == "nt") and (not output_dir.startswith("gs://"))
        winutils_issue = "winutils" in msg or "hadoop_home" in msg or "hadoop.home.dir" in msg
        if is_windows_local and winutils_issue:
            _write_csv_fallback(df, output_dir)
        else:
            raise


def save_recommendations_json(
    recommendations: DataFrame, output_path: str, k_value: int, sample_users: int = 20
) -> None:
    """
    Guarda las recomendaciones en dos archivos JSON:
      - recommendations_k{N}.json      : todos los usuarios agrupados por userId.
      - sample_20_users_k{N}.json      : muestra de los primeros 20 usuarios.

    Usa toLocalIterator() para transmitir filas al driver sin workers Python,
    y json.dump() nativo para escribir sin depender de HADOOP_HOME.
    """
    ordered = recommendations.orderBy("userId", "rank").toLocalIterator()

    by_user: Dict[str, List[Dict[str, object]]] = {}
    for row in ordered:
        uid = str(row["userId"])
        if uid not in by_user:
            by_user[uid] = []
        by_user[uid].append(
            {
                "rank": int(row["rank"]),
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "cluster": int(row["cluster"]),
                "cluster_avg_rating": round(float(row["cluster_avg_rating"]), 4),
                "cluster_num_ratings": int(row["cluster_num_ratings"]),
            }
        )

    all_file = os.path.join(output_path, f"recommendations_k{k_value}.json")
    with open(all_file, "w", encoding="utf-8") as handle:
        json.dump(by_user, handle, indent=2, ensure_ascii=False)

    sample_ids = sorted(by_user.keys(), key=int)[:sample_users]
    sample_data = {uid: by_user[uid] for uid in sample_ids}
    sample_file = os.path.join(output_path, f"sample_20_users_k{k_value}.json")
    with open(sample_file, "w", encoding="utf-8") as handle:
        json.dump(sample_data, handle, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote JSON: {all_file}")
    print(f"[INFO] Wrote JSON sample: {sample_file}")


def main() -> None:
    """
    Orquesta el pipeline completo:
      carga → EDA → split → features → clustering por K → recomendaciones
      → evaluación → guardado de JSONs y métricas.
    """
    args = parse_args()

    strategy = parse_strategy(args.strategy)
    positive_threshold = env_float("POSITIVE_RATING_THRESHOLD", 3.0)
    top_n = env_int("TOP_N_RECOMMENDATIONS", 10)
    min_cluster_votes = env_int("MIN_CLUSTER_RATINGS", 5)
    min_cluster_avg_rating = env_float("MIN_CLUSTER_AVG_RATING", 3.5)
    relevance_threshold = env_float("RELEVANCE_THRESHOLD", 4.0)
    shrinkage_beta = env_float("SCORE_SHRINKAGE_BETA", 25.0)
    weight_affinity = env_float("WEIGHT_AFFINITY", 0.45)
    weight_bayesian = env_float("WEIGHT_BAYESIAN", 0.45)
    weight_popularity = env_float("WEIGHT_POPULARITY", 0.10)
    k_values = parse_k_values(args.k_values)

    data_path = get_input_path(args.data_dir)
    output_path = get_output_path(args.output_path)

    spark = build_spark_session()

    try:
        print("=" * 72)
        print("MovieLens 1M - Cluster based recommendation lab")
        print("=" * 72)
        print(f"Input path        : {data_path}")
        print(f"Output path       : {output_path}")
        print(f"K values          : {k_values}")
        print(f"Strategy          : {strategy}")
        print(
            f"Params            : positive_threshold={positive_threshold}, top_n={top_n}, "
            f"min_cluster_votes={min_cluster_votes}, min_cluster_avg_rating={min_cluster_avg_rating}, "
            f"relevance_threshold={relevance_threshold}, shrinkage_beta={shrinkage_beta}, "
            f"weights=({weight_affinity},{weight_bayesian},{weight_popularity})"
        )

        if has_local_movielens_files(data_path):
            try:
                ratings, movies = load_movielens_data(spark, data_path)
            except Exception as exc:
                if args.demo_if_missing:
                    print(f"[WARN] Could not load data from {data_path}: {exc}")
                    print("[WARN] Using demo dataset")
                    ratings, movies = load_demo_data(spark)
                else:
                    raise
        else:
            if args.demo_if_missing:
                print("[WARN] ratings.dat/movies.dat not found. Using demo dataset")
                ratings, movies = load_demo_data(spark)
            else:
                raise FileNotFoundError(
                    f"ratings.dat or movies.dat not found in {data_path}. "
                    "Use --demo-if-missing for a local demo run."
                )

        ratings = ratings.cache()
        movies = movies.cache()
        exploratory_analysis(ratings, movies)

        # seed=42 garantiza reproducibilidad; randomSplit es aproximado (no exacto 80/20)
        train_ratings, test_ratings = ratings.randomSplit([0.8, 0.2], seed=42)
        train_ratings = train_ratings.cache()
        test_ratings = test_ratings.cache()

        train_count = train_ratings.count()
        test_count = test_ratings.count()
        total = train_count + test_count
        print("\n=== TRAIN / TEST SPLIT ===")
        print(
            f"Train: {train_count} ({100.0 * train_count / total:.1f}%) | "
            f"Test: {test_count} ({100.0 * test_count / total:.1f}%)"
        )

        # Solo datos de TRAIN para evitar data leakage en features y clustering
        user_genre_matrix, movie_genre_matrix, feature_cols = build_user_genre_features(
            train_ratings,
            movies,
            positive_threshold,
        )
        user_genre_matrix = user_genre_matrix.cache()
        movie_genre_matrix = movie_genre_matrix.cache()
        user_count = user_genre_matrix.count()
        # Media global de ratings en TRAIN: prior para el Bayesian smoothing
        global_mean_rating = float(train_ratings.agg(F.avg("rating").alias("avg")).first()["avg"])

        # K-Means requiere K < número de puntos; descartar K inválidos
        valid_k = [k for k in k_values if 2 <= k < user_count]
        if len(valid_k) < 2:
            raise ValueError(
                "Need at least 2 valid K values for comparison. "
                f"Provided={k_values}, valid={valid_k}, user_count={user_count}"
            )

        os.makedirs(output_path, exist_ok=True)

        print("\n=== EVALUATION BY K ===")
        results = []
        best_result = None

        for k in valid_k:
            print(f"\n--- K={k} ---")
            clusters, silhouette = run_kmeans_for_k(user_genre_matrix, k)
            clusters = clusters.cache()
            log_cluster_distribution(clusters, k)

            recommendations = generate_cluster_recommendations(
                train_ratings,
                clusters,
                user_genre_matrix,
                movie_genre_matrix,
                feature_cols,
                movies,
                top_n,
                min_cluster_votes,
                min_cluster_avg_rating,
                strategy,
                global_mean_rating,
                shrinkage_beta,
                weight_affinity,
                weight_bayesian,
                weight_popularity,
            ).cache()

            metrics = evaluate_recommendations(recommendations, test_ratings, relevance_threshold)

            print(
                f"silhouette={silhouette:.4f} | precision@{top_n}={metrics['precision']:.4f} | "
                f"recall@{top_n}={metrics['recall']:.4f}"
            )

            result = {
                "k": k,
                "silhouette": float(silhouette),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "num_users_evaluated": int(metrics["num_users_evaluated"]),
                "clusters": clusters,
                "recommendations": recommendations,
            }
            results.append(result)

            write_dataset(clusters, os.path.join(output_path, f"clusters_k{k}"))
            write_dataset(
                recommendations,
                os.path.join(output_path, f"recommendations_top{top_n}_k{k}"),
            )
            save_recommendations_json(recommendations, output_path, k_value=k, sample_users=20)

            if (best_result is None) or (result["precision"] > best_result["precision"]):
                best_result = result

        comparison_rows = [
            {
                "k": row["k"],
                "silhouette": round(row["silhouette"], 4),
                "precision_at_n": round(row["precision"], 4),
                "recall_at_n": round(row["recall"], 4),
                "num_users_evaluated": row["num_users_evaluated"],
            }
            for row in results
        ]

        metrics_payload = {
            "strategy": strategy,
            "top_n": top_n,
            "relevance_threshold": relevance_threshold,
            "k_values_evaluated": valid_k,
            "best_k_by_precision": best_result["k"],
            "results": comparison_rows,
        }

        metrics_file = os.path.join(output_path, "evaluation_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote JSON: {metrics_file}")

        print("\n=== COMPARISON TABLE ===")
        for row in comparison_rows:
            print(
                f"K={row['k']} | silhouette={row['silhouette']:.4f} | "
                f"precision@{top_n}={row['precision_at_n']:.4f} | "
                f"recall@{top_n}={row['recall_at_n']:.4f}"
            )

        print(f"[INFO] Best K by precision: {best_result['k']}")
        print("\n=== SAMPLE RECOMMENDATIONS (BEST K) ===")
        best_result["recommendations"].orderBy("userId", "rank").show(200, truncate=False)

        print("\n=== PIPELINE DONE ===")
        print(f"Outputs written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
