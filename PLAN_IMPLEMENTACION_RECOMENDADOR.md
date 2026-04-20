# Implementacion del plan de recomendacion

## Objetivo
Evolucionar el pipeline desde una suma de ratings positivos por genero hacia un recomendador mas robusto basado en:

1. Porcentaje de positivos por genero por usuario.
2. Calidad de peliculas a nivel de resenas.
3. Ranking personalizado Top-N por usuario.

---

## Paso 1: Redefinicion de features por genero (usuario)

### Que se cambio
Se reemplazo la metrica anterior (suma de ratings > 3) por una metrica de proporcion positiva por genero.

### Como se implemento
1. Se expanden las peliculas por genero con normalizacion por multi-genero (`1 / genre_count`).
2. Para cada usuario-genero se calculan:
   - `positive_weight`: suma ponderada de resenas positivas (`rating > 3`).
   - `total_weight`: suma ponderada de todas las resenas.
3. Se calcula la tasa positiva:

   `positive_rate = positive_weight / total_weight`

4. Se aplica suavizado bayesiano por genero para evitar ruido cuando hay pocos datos:

   `smoothed_rate = (positive_weight + alpha * global_positive_rate) / (total_weight + alpha)`

5. Se pivotea a matriz usuario x genero para usar en clustering y recomendacion.

### Beneficio
Las preferencias quedan en escala [0, 1], comparables entre usuarios y menos sensibles a datos escasos.

---

## Paso 2: Scoring de peliculas a nivel de reseñas

### Que se cambio
Se agrego una capa de calidad global de peliculas para mejorar el ranking.

### Como se implemento
Por pelicula se calculan:

1. `avg_rating`
2. `positive_ratio`
3. `rating_count`
4. `bayesian_rating`:

   $$bayesian\_rating = (avg\_rating * n + beta * global\_avg\_rating) / (n + beta)$$

5. `quality_score = bayesian_rating / 5`

Ademas, se genera `top_movies_by_genre` con ranking por genero usando calidad, ratio positiva y volumen de resenas.

### Beneficio
Se evita sobrevalorar peliculas con pocas resenas y se obtiene un ranking por genero mas estable.

---

## Paso 3: Recomendacion Top-N personalizada

### Que se cambio
Se agrego un rankeador hibrido usuario-pelicula.

### Como se implemento
1. Se convierte la matriz usuario-genero a formato largo (`userId`, `genre`, `genre_preference`).
2. Se calcula afinidad usuario-pelicula como promedio de preferencias de los generos de la pelicula.
3. Se excluyen peliculas ya vistas por el usuario.
4. Se calcula score final:

   `final_score = w_pref * affinity_score + w_quality * quality_score + w_positive * positive_ratio`

5. Se rankea por usuario y se guarda `recommendations_topN`.

### Beneficio
Combina gusto personal, calidad general y evidencia de reseñas positivas.

---

## Paso 4: Integracion en el pipeline principal

### Que se cambio
El flujo principal ahora ejecuta, en orden:

1. Carga y EDA.
2. Features de porcentaje positivo por genero (suavizado).
3. Clustering K-Means para segmentacion.
4. Perfil de calidad de peliculas por reseñas.
5. Generacion de recomendaciones Top-N.

### Salidas generadas
En `output_path` se escriben:

1. `clusters_k{best_k}`
2. `movie_quality_scores`
3. `top_movies_by_genre`
4. `recommendations_top{N}`

---

## Parametros configurables (variables de entorno)

1. `GENRE_SMOOTHING_ALPHA` (default: 5)
2. `MOVIE_BAYES_BETA` (default: 20)
3. `TOP_MOVIES_PER_GENRE` (default: 10)
4. `TOP_N_RECOMMENDATIONS` (default: 10)
5. `WEIGHT_PREFERENCE` (default: 0.6)
6. `WEIGHT_QUALITY` (default: 0.3)
7. `WEIGHT_POSITIVE` (default: 0.1)

---

## Nota de uso
El script mantiene el clustering (segmentacion) y agrega una capa completa de recomendacion de peliculas basada en preferencias por genero y senales a nivel de resenas.
