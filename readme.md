# Sistema de Recomendación Basado en Clusters de Usuarios

**Laboratorio Semana 9 — Sistemas Distribuidos de Datos**
Universidad ICESI

## Integrantes

- Ricardo Andrés Chamorro Martinez
- Diego Armando Polanco Lozano

---

## Descripción

Sistema de recomendación de películas construido con **Apache Spark** sobre el dataset **MovieLens 1M**. Los usuarios se agrupan mediante **K-Means clustering** según sus preferencias de género, y se recomiendan películas no vistas usando la estrategia **Confidence Hybrid** que combina afinidad personal, rating Bayesiano y popularidad.

---

## Dataset

| Campo | Valor |
|---|---|
| Dataset | MovieLens 1M |
| Ratings | 1,000,209 |
| Usuarios | 6,040 |
| Películas | 3,883 |
| Densidad | 4.2647% |
| Rating promedio | 3.58 |
| Features de género | 18 |

---

## Estructura del Proyecto

```
semana9-recommendationsystem/
│
├── spark-kmeans-local.py     # Pipeline completo para ejecución local (Windows)
├── spark-kmeans.py           # Pipeline para despliegue en GCP / Spark cluster
├── recommendation_api.py     # API HTTP para consultar recomendaciones
│
├── ml-1m/                    # Dataset MovieLens 1M
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
│
└── output/                   # Resultados generados
    ├── recommendations_k3.json
    ├── recommendations_k5.json
    ├── recommendations_k8.json
    ├── recommendations_k10.json
    ├── sample_20_users_k{N}.json
    ├── evaluation_metrics.json
    └── diagrama_estrategia.html
```

---

## Estrategia de Recomendación

### Pipeline

```
Dataset → Split 80/20 → Features → K-Means → Candidatos → Scoring → Top-10 → Evaluación
```

### 1. Construcción de Features

- **Features de usuario** (`user_genre_matrix`): para cada usuario se suman los `weighted_rating` por género, donde `weighted_rating = rating / número_de_géneros`. Solo se usan ratings positivos (`rating > 3.0`).
- **Features de película** (`movie_genre_matrix`): presencia binaria de cada género por película.

### 2. Logs del K-Means

Durante cada corrida se imprime:

- Silhouette score por cada `K`.
- Cantidad de usuarios por cluster y porcentaje sobre el total.
- Resumen rápido con mínimo, máximo y promedio de tamaño de cluster.

### 3. Clustering K-Means

| Paso | Detalle |
|---|---|
| Normalización | StandardScaler (media=0, std=1) |
| Algoritmo | K-Means, seed=42, maxIter=40 |
| K evaluados | 3, 5, 8, 10 |
| Métrica de calidad | Silhouette Score |

### 4. Generación de Candidatos

- Se toman las películas valoradas por el cluster del usuario.
- Se excluyen películas ya vistas en **Train** mediante `left_anti join`.
- Filtros de calidad: `votos ≥ 5` y `avg_rating ≥ 3.5` en el cluster.

### 5. Confidence Hybrid Score

El score final combina tres componentes normalizados:

```
ranking_score = 0.45 × affinity_norm
              + 0.45 × (bayesian_rating / 5.0)
              + 0.10 × popularity_norm
```

| Componente | Descripción |
|---|---|
| **Affinity** | Producto punto entre vector de géneros del usuario y de la película, normalizado a [0,1] por usuario |
| **Bayesian Rating** | `(avg_cluster × n + μ_global × β) / (n + β)` con β=25. Suaviza el promedio del cluster con la media global para películas con pocos votos |
| **Popularity** | `log(votos) / log(max_votos)` — popularidad relativa en escala logarítmica |

---

## Resultados

### Métricas por Configuración de K

| K | Silhouette | Precision@10 | Recall@10 |
|---|---|---|---|
| K=3  | 0.5617 | 0.0932 | 0.0679 |
| K=5  | 0.4210 | 0.0973 | 0.0702 |
| K=8  | 0.4154 | 0.0999 | 0.0710 |
| K=10 | 0.2760 | 0.1017 | 0.0726 |

**Mejor configuración por Precision@10: K=10**

> Los valores de Precision@10 ~10% son ~11× mejores que un recomendador aleatorio (~0.9%), lo cual es un resultado sólido para un sistema cluster-based CF sobre MovieLens 1M.

### Ejemplo de Recomendaciones (Usuario 1, K=10, Cluster 0)

| Rank | Película | Score |
|---|---|---|
| 1 | This Is Spinal Tap (1984) | 4.26 |
| 2 | Babe (1995) | 3.894 |
| 3 | Watership Down (1978) | 3.853 |
| 4 | Star Wars: Episode V - The Empire Strikes Back (1980) | 4.178 |
| 5 | American Beauty (1999) | 4.343 |
| 6 | Cinema Paradiso (1988) | 4.402 |
| 7 | Stand by Me (1986) | 4.135 |
| 8 | Life Is Beautiful (1997) | 4.365 |
| 9 | Lethal Weapon (1987) | 3.797 |
| 10 | Godfather, The (1972) | 4.479 |

---

## Ejecución

### Local (Windows)

```bash
python spark-kmeans-local.py
```

Requisitos: Python 3.x, PySpark, Java 8+. No requiere Hadoop ni winutils.

### API HTTP

```bash
python recommendation_api.py --host 127.0.0.1 --port 8000
```

Abre el front en `http://127.0.0.1:8000/` y el endpoint de salud en `http://127.0.0.1:8000/health`.

Ejemplos de uso:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/recommendations ^
    -H "Content-Type: application/json" ^
    -d "{\"userId\": 1, \"top_n\": 10}"
```

```bash
curl -X POST http://127.0.0.1:8000/recommendations ^
    -H "Content-Type: application/json" ^
    -d "{\"movieIds\": [1, 260, 1196], \"ratings\": {\"1\": 5, \"260\": 4}, \"top_n\": 10}"
```

La API responde con una de estas dos modalidades:

- `userId`: devuelve las recomendaciones cluster-based precalculadas del usuario.
- `movieIds`: construye un perfil temporal con las películas marcadas y devuelve recomendaciones por afinidad de género, calidad y popularidad.

En el modo `userId`, la respuesta también incluye:

- cluster asignado al usuario
- matriz de gustos del usuario
- matriz promedio de gustos del cluster
- detalle de ratings por película recomendada

### GCP / Spark Cluster

```bash
export SPARK_MASTER_URL=spark://<master-ip>:7077
export MOVIELENS_GCS_BUCKET=gs://mi-bucket
python spark-kmeans.py --k-values 3,5,8,10 --strategy confidence_hybrid
```

---

## Evaluación

- **Precision@10**: fracción de las 10 recomendaciones que el usuario calificó con ≥ 4 en el conjunto de Test.
- **Recall@10**: fracción de todos los ítems relevantes del usuario en Test que fueron recomendados.
- **Silhouette Score**: mide la cohesión y separación de los clusters (valores cercanos a 1 = mejor clustering).
- La evaluación usa **inner join**: solo usuarios con al menos un ítem relevante en Test son incluidos en el promedio.

---

## Limitaciones

- El enfoque cluster-based asigna recomendaciones similares a todos los usuarios del mismo cluster. El scoring de afinidad individual mitiga esto pero no lo elimina.
- `randomSplit([0.8, 0.2])` en Spark es aproximado (puede mostrar 79.9% / 20.1%), lo cual es comportamiento normal.
- K-Means asume clusters convexos y puede generar clusters muy desiguales en tamaño (ej. K=10 genera un cluster con ~44% de usuarios).
