"""
Sistema de Recomendación Basado en Clusters de Usuarios — Versión LOCAL (Windows)
==================================================================================
Dataset  : MovieLens 1M (ratings.dat, movies.dat)
Objetivo : Agrupar usuarios con K-Means según sus preferencias de género y
           recomendar películas no vistas usando la estrategia Confidence Hybrid.

Pasos del pipeline:
  1. Cargar y explorar datos (ratings + películas)
  2. División Train 80% / Test 20%
  3. Construir features de usuario y película por género
  4. Clustering K-Means con K ∈ {3, 5, 8, 10}
  5. Generar candidatos (películas del cluster no vistas en Train)
  6. Puntuar candidatos con Confidence Hybrid (afinidad + Bayesian + popularidad)
  7. Evaluar con Precision@10 y Recall@10 sobre el conjunto de Test
  8. Guardar recomendaciones en JSON
"""

import os
import sys
import json
import math
from functools import reduce
from typing import Dict, List, Tuple

# ── Corrección de versión Python en Windows ──────────────────────────────────
# PySpark lanza workers Python independientes; si hay varias versiones
# instaladas, el worker puede usar una distinta al driver y fallar.
# Forzar ambos a usar el mismo ejecutable que corre este script lo evita.
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, LongType, StringType, StructField, StructType

# ── Rutas de datos y salida ───────────────────────────────────────────────────
# Se resuelven relativas al directorio del script para que funcione
# independientemente de desde dónde se ejecute.
DATA_PATH   = os.path.join(os.path.dirname(__file__), "ml-1m")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output")

# ── Configuración del experimento ─────────────────────────────────────────────
K_VALUES  = [3, 5, 8, 10]          # Valores de K a evaluar
STRATEGY  = "confidence_hybrid"    # Estrategia de scoring
TOP_N     = 10                     # Recomendaciones top-N por usuario

# ── Parámetros de calidad y scoring ──────────────────────────────────────────
POSITIVE_THRESHOLD  = 3.0   # Rating mínimo para considerar un rating como positivo al construir features
MIN_CLUSTER_VOTES   = 5     # Mínimo de votos de usuarios del cluster para incluir una película
MIN_CLUSTER_AVG     = 3.5   # Rating promedio mínimo en el cluster para incluir una película
RELEVANCE_THRESHOLD = 4.0   # Rating mínimo en TEST para considerar un ítem como relevante
SHRINKAGE_BETA      = 25.0  # Factor de contracción Bayesiana: cuánto "peso" tiene la media global
WEIGHT_AFFINITY     = 0.45  # Peso del componente de afinidad de género en el score final
WEIGHT_BAYESIAN     = 0.45  # Peso del rating Bayesiano en el score final
WEIGHT_POPULARITY   = 0.10  # Peso de la popularidad en el score final


def build_spark_session() -> SparkSession:
    """Crea y devuelve una SparkSession configurada para ejecución local."""
    return (
        SparkSession.builder
        .appName("MovieLens_RecommendationSystem_Local")
        .master("local[*]")                              # Usa todos los cores disponibles
        .config("spark.sql.shuffle.partitions", "8")     # Reduce particiones de shuffle para datos pequeños
        .config("spark.default.parallelism", "8")        # Paralelismo por defecto
        .config("spark.sql.debug.maxToStringFields", "200")
        .getOrCreate()
    )


def load_data(spark: SparkSession) -> Tuple[DataFrame, DataFrame]:
    """
    Carga ratings.dat y movies.dat desde DATA_PATH.
    El separador '::' es el formato nativo de MovieLens 1M.
    Se define el schema explícitamente para evitar el costoso inferSchema.
    """
    ratings_schema = StructType([
        StructField("userId",    IntegerType(), False),
        StructField("movieId",   IntegerType(), False),
        StructField("rating",    FloatType(),   False),
        StructField("timestamp", LongType(),    False),
    ])
    movies_schema = StructType([
        StructField("movieId", IntegerType(), False),
        StructField("title",   StringType(),  True),
        StructField("genres",  StringType(),  True),  # Géneros separados por '|'
    ])
    df_ratings = (spark.read
                  .option("delimiter", "::")
                  .schema(ratings_schema)
                  .csv(os.path.join(DATA_PATH, "ratings.dat")))
    df_movies = (spark.read
                 .option("delimiter", "::")
                 .schema(movies_schema)
                 .csv(os.path.join(DATA_PATH, "movies.dat")))
    return df_ratings, df_movies


def exploratory_analysis(ratings: DataFrame, movies: DataFrame) -> None:
    """
    Imprime estadísticas básicas del dataset: usuarios únicos, películas,
    total de ratings, densidad de la matriz usuario-película y distribución
    de ratings. Útil para verificar que los datos se cargaron correctamente.
    """
    users         = ratings.select("userId").distinct().count()
    movie_count   = movies.count()
    ratings_count = ratings.count()
    print("\n=== EDA ===")
    print(f"Users      : {users:,}")
    print(f"Movies     : {movie_count:,}")
    print(f"Ratings    : {ratings_count:,}")
    # Densidad: % de pares (usuario, película) que tienen rating
    print(f"Density    : {100 * ratings_count / (users * movie_count):.4f}%")
    print("Ratings distribution:")
    ratings.groupBy("rating").count().orderBy("rating").show(truncate=False)
    avg_r = ratings.agg(F.avg("rating")).first()[0]
    print(f"Avg rating : {avg_r:.2f}")


def safe_genre_column(genre_name: str) -> str:
    """
    Convierte un nombre de género a un nombre de columna Spark seguro.
    Reemplaza caracteres especiales que causan errores de parsing en columnas.
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
    Construye dos matrices de features:
      - user_genre_matrix : userId × género → suma de weighted_rating
        weighted_rating = rating / número_de_géneros_de_la_película
        Solo ratings positivos (> positive_threshold) para capturar preferencias reales.
      - movie_genre_matrix: movieId × género → presencia binaria (1.0 si el género existe)

    La ponderación por número de géneros evita que películas con muchos géneros
    acumulen más puntuación que películas de un solo género.

    Retorna (user_genre_matrix, movie_genre_matrix, feature_cols).
    """
    # Expandir la columna 'genres' (ej. "Action|Thriller") en filas individuales
    movies_expanded = (
        movies.withColumn("genre_array", F.split(F.col("genres"), "\\|"))
        .withColumn("genre_count", F.size(F.col("genre_array")))
        .select("movieId", "genre_count", F.explode(F.col("genre_array")).alias("genre"))
    )

    # Filtrar solo ratings positivos para construir el perfil de usuario
    positive = train_ratings.filter(F.col("rating") > F.lit(positive_threshold))

    # Unir ratings con géneros y calcular el rating ponderado por película
    train_genre = (
        positive.join(movies_expanded, "movieId", "inner")
        .withColumn("weighted_rating", F.col("rating") / F.col("genre_count"))
    )

    # Lista de géneros únicos en el dataset (se usa como eje del pivot)
    genre_values = [
        row["genre"]
        for row in movies_expanded.select("genre").distinct()
        .where(F.col("genre").isNotNull()).collect()
    ]

    # Pivot: cada fila = usuario, cada columna = género, valor = suma de weighted_ratings
    user_genre_matrix = (
        train_genre.groupBy("userId")
        .pivot("genre", genre_values)
        .agg(F.sum("weighted_rating"))
        .fillna(0.0)  # Usuarios sin ratings en algún género reciben 0
    )

    # Pivot: cada fila = película, cada columna = género, valor = 1.0 si pertenece al género
    movie_genre_matrix = (
        movies_expanded.groupBy("movieId")
        .pivot("genre", genre_values)
        .agg(F.count("genre").cast("double"))
        .fillna(0.0)
    )

    # Renombrar columnas a nombres seguros y garantizar que existan todos los géneros
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
      1. VectorAssembler: convierte columnas de género en un vector denso.
      2. StandardScaler: normaliza cada feature (media=0, std=1) para que géneros
         con mayor varianza no dominen la distancia euclidiana del clustering.
      3. KMeans: agrupa usuarios en k clusters.
      4. ClusteringEvaluator: calcula el Silhouette Score para medir cohesión
         interna (valores cercanos a 1 = clusters bien separados).

    Retorna (DataFrame con userId y cluster asignado, silhouette score).
    """
    feature_cols = [c for c in user_genre_matrix.columns if c != "userId"]

    # Paso 1: ensamblar features en un vector
    assembled = VectorAssembler(inputCols=feature_cols, outputCol="features_raw").transform(
        user_genre_matrix
    )

    # Paso 2: escalar features para normalizar la importancia de cada género
    scaled = (
        StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        .fit(assembled)
        .transform(assembled)
    )

    # Paso 3: entrenar K-Means y asignar cluster a cada usuario
    model = KMeans(k=k_value, featuresCol="features", predictionCol="prediction",
                   seed=42, maxIter=40).fit(scaled)
    predictions = model.transform(scaled)

    # Paso 4: evaluar calidad del clustering con Silhouette Score
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
    Genera recomendaciones Top-N para cada usuario usando su cluster.

    Flujo:
      1. Calcular estadísticas de rating por (cluster, película).
      2. Filtrar películas con suficientes votos y rating aceptable en el cluster.
      3. Excluir películas ya vistas por el usuario en Train (left_anti join).
      4. Calcular Affinity Score: producto punto entre el vector de géneros del
         usuario y el vector binario de géneros de la película.
      5. Si strategy == 'confidence_hybrid':
           - Normalizar afinidad por usuario a [0, 1].
           - Calcular Bayesian Rating: suaviza el promedio del cluster con la
             media global para evitar sobreestimación en películas con pocos votos.
           - Calcular Popularity Norm: log(votos) normalizado para premiar
             películas con más respaldo del cluster.
           - Score final = 0.45*affinity + 0.45*bayesian + 0.10*popularity
      6. Rankear por usuario y tomar Top-N.
    """
    # Unir ratings de train con la asignación de cluster de cada usuario
    train_clustered = train_ratings.join(clusters, "userId", "inner")

    # Calcular el rating promedio y número de votos de cada película por cluster
    cluster_movie_scores = (
        train_clustered.groupBy("cluster", "movieId")
        .agg(
            F.avg("rating").alias("cluster_avg_rating"),
            F.count("*").alias("cluster_num_ratings"),
        )
        # Filtro de calidad: evitar películas con poca evidencia o mal valoradas
        .filter(
            (F.col("cluster_num_ratings") >= F.lit(min_cluster_votes))
            & (F.col("cluster_avg_rating") >= F.lit(min_cluster_avg_rating))
        )
    )

    # Películas ya vistas por cada usuario en Train (a excluir de las recomendaciones)
    seen = train_ratings.select("userId", "movieId").distinct()

    # Candidatos: películas del cluster del usuario que NO ha visto en Train
    candidates = clusters.join(cluster_movie_scores, "cluster", "inner")
    candidates = candidates.join(seen, ["userId", "movieId"], "left_anti")

    # ── Affinity Score ─────────────────────────────────────────────────────────
    # Renombrar columnas para evitar ambigüedad en el join
    user_features = user_genre_matrix.select(
        "userId", *[F.col(c).alias(f"u_{c}") for c in feature_cols]
    )
    movie_features = movie_genre_matrix.select(
        "movieId", *[F.col(c).alias(f"m_{c}") for c in feature_cols]
    )
    # Producto punto: suma de (preferencia_usuario_género × presencia_película_género)
    affinity_expr = reduce(
        lambda a, b: a + b,
        [F.col(f"u_{c}") * F.col(f"m_{c}") for c in feature_cols],
    )
    candidates = (
        candidates.join(user_features, "userId", "inner")
        .join(movie_features, "movieId", "inner")
        .withColumn("affinity_score", affinity_expr)
        .withColumn("ranking_score", F.col("affinity_score"))  # Score base = afinidad
    )

    # ── Confidence Hybrid Score ────────────────────────────────────────────────
    if strategy == "confidence_hybrid":
        # Calcular mínimo y máximo de afinidad por usuario para normalizar a [0,1]
        user_affinity_bounds = candidates.groupBy("userId").agg(
            F.min("affinity_score").alias("affinity_min"),
            F.max("affinity_score").alias("affinity_max"),
        )
        # Máximo de votos global para normalizar la popularidad
        max_votes = float(
            cluster_movie_scores.agg(F.max("cluster_num_ratings").alias("m")).first()["m"] or 1.0
        )

        candidates = candidates.join(user_affinity_bounds, "userId", "left")

        # Normalización min-max de la afinidad por usuario
        # Si min == max (todos los candidatos tienen igual afinidad), se asigna 0.5
        candidates = candidates.withColumn(
            "affinity_norm",
            F.when(
                F.col("affinity_max") > F.col("affinity_min"),
                (F.col("affinity_score") - F.col("affinity_min"))
                / (F.col("affinity_max") - F.col("affinity_min")),
            ).otherwise(F.lit(0.5)),
        )

        # Bayesian Average: (avg_cluster * n + global_mean * beta) / (n + beta)
        # Beta controla cuánto "peso" tiene la media global respecto a los votos del cluster.
        # Con pocos votos (n << beta), el rating se acerca a la media global;
        # con muchos votos (n >> beta), el rating se acerca al promedio del cluster.
        candidates = candidates.withColumn(
            "bayesian_rating",
            (F.col("cluster_avg_rating") * F.col("cluster_num_ratings")
             + F.lit(global_mean_rating) * F.lit(shrinkage_beta))
            / (F.col("cluster_num_ratings") + F.lit(shrinkage_beta)),
        )

        # Popularidad normalizada con escala logarítmica para reducir el impacto
        # de diferencias muy grandes entre películas populares y poco vistas
        pop_denom = float(math.log1p(max_votes)) or 1.0
        candidates = candidates.withColumn(
            "popularity_norm",
            F.log1p(F.col("cluster_num_ratings")) / F.lit(pop_denom),
        )

        # Score final: combinación ponderada de los tres componentes
        candidates = candidates.withColumn(
            "ranking_score",
            F.lit(weight_affinity) * F.col("affinity_norm")
            + F.lit(weight_bayesian) * (F.col("bayesian_rating") / F.lit(5.0))
            + F.lit(weight_popularity) * F.col("popularity_norm"),
        )

    # ── Ranking Top-N por usuario ──────────────────────────────────────────────
    # Window function: particionar por usuario y ordenar por score descendente
    rank_window = Window.partitionBy("userId").orderBy(
        F.desc("ranking_score"),
        F.desc("affinity_score"),       # Desempate 1: mayor afinidad personal
        F.desc("cluster_avg_rating"),   # Desempate 2: mejor rating en el cluster
        F.desc("cluster_num_ratings"),  # Desempate 3: más votos en el cluster
        F.asc("movieId"),               # Desempate final: reproducibilidad
    )
    recs = (
        candidates.withColumn("rank", F.row_number().over(rank_window))
        .filter(F.col("rank") <= top_n)
        .join(movies.select("movieId", "title", "genres"), "movieId", "left")
    )
    return recs.select(
        "userId", "cluster", "movieId", "title", "genres",
        "ranking_score", "affinity_score", "cluster_avg_rating", "cluster_num_ratings", "rank",
    )


def evaluate_recommendations(
    recommendations: DataFrame, test_ratings: DataFrame, relevance_threshold: float
) -> Dict[str, float]:
    """
    Evalúa las recomendaciones con Precision@N y Recall@N sobre el conjunto de Test.

    Definiciones:
      - Relevante: ítem calificado en TEST con rating >= relevance_threshold.
      - Hit: película recomendada que el usuario también calificó como relevante en TEST.
      - Precision@N = hits / N  (fracción de las N recomendaciones que son relevantes)
      - Recall@N    = hits / total_relevantes  (fracción de relevantes recuperados)

    Se usa inner join para evaluar solo usuarios que tienen ítems relevantes en TEST,
    evitando que usuarios sin relevantes arrastren el promedio hacia cero.
    """
    # Ítems relevantes en TEST (rating >= umbral)
    relevant = (
        test_ratings.filter(F.col("rating") >= F.lit(relevance_threshold))
        .select("userId", "movieId")
        .withColumn("is_relevant", F.lit(1.0))
    )

    # Contar hits y total de recomendaciones por usuario
    user_hits = (
        recommendations.select("userId", "movieId")
        .join(relevant, ["userId", "movieId"], "left")   # left: los no-hits quedan como 0
        .fillna(0.0, subset=["is_relevant"])
        .groupBy("userId")
        .agg(F.sum("is_relevant").alias("num_hits"), F.count("*").alias("num_recs"))
    )

    # Total de ítems relevantes por usuario en TEST
    rel_per_user = relevant.groupBy("userId").agg(F.count("*").alias("num_relevant"))

    # Inner join: solo usuarios con al menos un ítem relevante en TEST
    eval_df = (
        user_hits.join(rel_per_user, "userId", "inner")
        .withColumn("precision", F.col("num_hits") / F.col("num_recs"))
        .withColumn("recall",    F.col("num_hits") / F.col("num_relevant"))
    )

    agg = eval_df.agg(
        F.avg("precision").alias("avg_precision"),
        F.avg("recall").alias("avg_recall"),
        F.count("*").alias("num_users_evaluated"),
    ).first()
    return {
        "precision":           float(agg["avg_precision"] or 0.0),
        "recall":              float(agg["avg_recall"]    or 0.0),
        "num_users_evaluated": int(agg["num_users_evaluated"]),
    }


def save_recommendations_json(
    recommendations: DataFrame, output_path: str, k_value: int, sample_users: int = 20
) -> None:
    """
    Guarda las recomendaciones en dos archivos JSON:
      - recommendations_k{N}.json      : todos los usuarios
      - sample_20_users_k{N}.json      : muestra de los primeros 20 usuarios

    Usa toLocalIterator() y json.dump() nativos de Python en lugar de
    df.write.json() de Spark, lo cual requeriría winutils.exe en Windows.
    """
    by_user: Dict[str, list] = {}
    # toLocalIterator() transmite filas al driver sin usar workers Python
    for row in recommendations.orderBy("userId", "rank").toLocalIterator():
        uid = str(row["userId"])
        if uid not in by_user:
            by_user[uid] = []
        by_user[uid].append({
            "rank":                int(row["rank"]),
            "movieId":             int(row["movieId"]),
            "title":               row["title"],
            "genres":              row["genres"],
            "cluster":             int(row["cluster"]),
            "cluster_avg_rating":  round(float(row["cluster_avg_rating"]), 4),
            "cluster_num_ratings": int(row["cluster_num_ratings"]),
        })

    os.makedirs(output_path, exist_ok=True)

    # Archivo completo con todos los usuarios
    all_file = os.path.join(output_path, f"recommendations_k{k_value}.json")
    with open(all_file, "w", encoding="utf-8") as fh:
        json.dump(by_user, fh, indent=2, ensure_ascii=False)

    # Archivo de muestra con los primeros N usuarios (para verificación rápida)
    sample_ids  = sorted(by_user.keys(), key=int)[:sample_users]
    sample_file = os.path.join(output_path, f"sample_20_users_k{k_value}.json")
    with open(sample_file, "w", encoding="utf-8") as fh:
        json.dump({uid: by_user[uid] for uid in sample_ids}, fh, indent=2, ensure_ascii=False)

    print(f"Recomendaciones (K={k_value}) guardadas en  : {all_file}")
    print(f"Muestra 20 usuarios guardada en             : {sample_file}")


def main() -> None:
    """Orquesta todo el pipeline: carga → split → features → clustering → recomendaciones → evaluación."""
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")  # Suprimir logs INFO/DEBUG de Spark

    try:
        print(f"\n{'='*72}")
        print("MovieLens 1M - Cluster based recommendation lab (LOCAL)")
        print(f"{'='*72}")
        print(f"Input path  : {DATA_PATH}")
        print(f"Output path : {OUTPUT_PATH}")
        print(f"K values    : {K_VALUES}")
        print(f"Strategy    : {STRATEGY}")
        print(
            f"Params      : positive_threshold={POSITIVE_THRESHOLD}, top_n={TOP_N}, "
            f"min_cluster_votes={MIN_CLUSTER_VOTES}, min_cluster_avg={MIN_CLUSTER_AVG}, "
            f"relevance_threshold={RELEVANCE_THRESHOLD}, shrinkage_beta={SHRINKAGE_BETA}, "
            f"weights=({WEIGHT_AFFINITY},{WEIGHT_BAYESIAN},{WEIGHT_POPULARITY})"
        )

        # ── Carga de datos ─────────────────────────────────────────────────────
        df_ratings, df_movies = load_data(spark)
        df_ratings = df_ratings.cache()  # Cachear para reutilizar en split y EDA
        df_movies  = df_movies.cache()
        exploratory_analysis(df_ratings, df_movies)

        # ── División Train / Test ──────────────────────────────────────────────
        # seed=42 garantiza reproducibilidad; randomSplit es aproximado (no exacto 80/20)
        df_train, df_test = df_ratings.randomSplit([0.8, 0.2], seed=42)
        df_train = df_train.cache()
        df_test  = df_test.cache()

        train_count = df_train.count()
        test_count  = df_test.count()
        total = train_count + test_count
        print("\n=== TRAIN / TEST SPLIT ===")
        print(
            f"Train: {train_count:,} ({100.0 * train_count / total:.1f}%) | "
            f"Test: {test_count:,} ({100.0 * test_count / total:.1f}%)"
        )

        # ── Construcción de features ───────────────────────────────────────────
        # Solo se usan datos de TRAIN para evitar data leakage
        user_genre_matrix, movie_genre_matrix, feature_cols = build_user_genre_features(
            df_train, df_movies, POSITIVE_THRESHOLD
        )
        user_genre_matrix  = user_genre_matrix.cache()
        movie_genre_matrix = movie_genre_matrix.cache()

        user_count  = user_genre_matrix.count()
        # Media global de ratings en TRAIN (usada como prior en Bayesian smoothing)
        global_mean = float(df_train.agg(F.avg("rating").alias("avg")).first()["avg"])
        # Excluir K >= user_count (K-Means requiere K < número de puntos)
        valid_k     = [k for k in K_VALUES if 2 <= k < user_count]

        print(f"\nUsuarios con features  : {user_count:,}")
        print(f"Feature cols           : {len(feature_cols)}")
        print(f"Rating promedio (train): {global_mean:.4f}")

        # ── Bucle principal: evaluar cada configuración de K ───────────────────
        print("\n=== EVALUATION BY K ===")
        results     = []
        best_result = None

        for k in valid_k:
            print(f"\n--- K={k} ---")

            # Clustering: agrupar usuarios en k clusters según sus preferencias
            clusters, silhouette = run_kmeans_for_k(user_genre_matrix, k)
            clusters = clusters.cache()
            log_cluster_distribution(clusters, k)

            # Generación y scoring de recomendaciones usando TRAIN
            recommendations = generate_cluster_recommendations(
                df_train, clusters, user_genre_matrix, movie_genre_matrix,
                feature_cols, df_movies, TOP_N,
                MIN_CLUSTER_VOTES, MIN_CLUSTER_AVG, STRATEGY, global_mean,
                SHRINKAGE_BETA, WEIGHT_AFFINITY, WEIGHT_BAYESIAN, WEIGHT_POPULARITY,
            ).cache()

            # Evaluación de métricas sobre TEST
            metrics = evaluate_recommendations(recommendations, df_test, RELEVANCE_THRESHOLD)
            print(
                f"silhouette={silhouette:.4f} | "
                f"precision@{TOP_N}={metrics['precision']:.4f} | "
                f"recall@{TOP_N}={metrics['recall']:.4f}"
            )

            result = {
                "k":                   k,
                "silhouette":          float(silhouette),
                "precision":           float(metrics["precision"]),
                "recall":              float(metrics["recall"]),
                "num_users_evaluated": int(metrics["num_users_evaluated"]),
                "clusters":            clusters,
                "recommendations":     recommendations,
            }
            results.append(result)
            save_recommendations_json(recommendations, OUTPUT_PATH, k_value=k)

            # Actualizar mejor configuración según Precision@N
            if best_result is None or result["precision"] > best_result["precision"]:
                best_result = result

        # ── Tabla comparativa y guardado de métricas ───────────────────────────
        comparison_rows = [
            {
                "k":                   r["k"],
                "silhouette":          round(r["silhouette"], 4),
                "precision_at_n":      round(r["precision"],  4),
                "recall_at_n":         round(r["recall"],     4),
                "num_users_evaluated": r["num_users_evaluated"],
            }
            for r in results
        ]

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        metrics_file = os.path.join(OUTPUT_PATH, "evaluation_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "strategy":            STRATEGY,
                    "top_n":               TOP_N,
                    "relevance_threshold": RELEVANCE_THRESHOLD,
                    "k_values_evaluated":  valid_k,
                    "best_k_by_precision": best_result["k"],
                    "results":             comparison_rows,
                },
                fh, indent=2, ensure_ascii=False,
            )
        print(f"\nMétricas de evaluación guardadas en: {metrics_file}")

        print("\n=== COMPARISON TABLE ===")
        for row in comparison_rows:
            print(
                f"K={row['k']} | silhouette={row['silhouette']:.4f} | "
                f"precision@{TOP_N}={row['precision_at_n']:.4f} | "
                f"recall@{TOP_N}={row['recall_at_n']:.4f}"
            )
        print(f"\n→ Mejor K por Precision@{TOP_N}: K={best_result['k']}")

        # ── Muestra de recomendaciones del mejor K ─────────────────────────────
        print(f"\n=== SAMPLE RECOMMENDATIONS (K={best_result['k']}, primeros 20 usuarios) ===")
        # .collect() corre en el driver — evita version mismatch con workers Python
        first_20 = [r[0] for r in
                    best_result["recommendations"].select("userId").distinct()
                    .orderBy("userId").limit(20).collect()]
        (best_result["recommendations"]
         .filter(F.col("userId").isin(first_20))
         .orderBy("userId", "rank")
         .select("userId", "cluster", "rank", "movieId", "title",
                 F.round("cluster_avg_rating", 3).alias("score"))
         .show(200, truncate=False))

        print(f"\n{'='*72}")
        print("RESUMEN FINAL")
        print(f"{'='*72}")
        print(f"Dataset      : MovieLens 1M (local)")
        print(f"Train / Test : {train_count:,} / {test_count:,} ratings")
        for r in results:
            print(
                f"  K={r['k']:<3} → Silhouette={r['silhouette']:.4f}  "
                f"Precision@{TOP_N}={r['precision']:.4f}  Recall@{TOP_N}={r['recall']:.4f}"
            )
        print(f"Mejor K      : {best_result['k']}")
        print(f"JSONs en     : {OUTPUT_PATH}")
        print(f"{'='*72}\n")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
