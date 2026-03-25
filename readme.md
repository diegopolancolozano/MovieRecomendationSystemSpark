Ricardo Andres Chamorro Martinez A003999846
Diego Armando Polanco Lozano A00399926

## Estructura actual (sin K-Means)

- `spark_master.py`: driver/master logico del proyecto. Crea sesion Spark, carga datos, analiza y guarda features.
- `spark_worker.py`: transformaciones distribuidas (feature engineering y escalado).
- `spark-kmeans.py`: entrypoint compatible que delega a `spark_master.py`.

## Levantar Spark Standalone (1 master + 2 workers)

> Reemplaza `SPARK_HOME` y `IP_MASTER` por tus valores reales.

### 1) Iniciar master

En la maquina master:

```powershell
& "$env:SPARK_HOME\sbin\start-master.cmd"
```

### 2) Iniciar worker 1

```powershell
& "$env:SPARK_HOME\sbin\start-worker.cmd" spark://IP_MASTER:7077
```

### 3) Iniciar worker 2

```powershell
& "$env:SPARK_HOME\sbin\start-worker.cmd" spark://IP_MASTER:7077
```

## Ejecutar pipeline (sin K-Means)

Desde este repositorio:

```powershell
$env:SPARK_MASTER_URL = "spark://IP_MASTER:7077"
$env:MOVIELENS_INPUT_BASE_PATH = "gs://tu-bucket/ml-100k"   # o ruta local compartida
$env:MOVIELENS_OUTPUT_BASE_PATH = "gs://tu-bucket/output"    # o ruta local compartida

python spark_master.py
```

Spark escribira las salidas en carpetas con archivos `part-*.csv`.

## Despliegue en GCP con bucket

Si vas a correrlo en GCP, puedes usar una sola variable para el bucket.

```powershell
$env:SPARK_MASTER_URL = "spark://IP_MASTER:7077"
$env:MOVIELENS_GCS_BUCKET = "tu-bucket"

python spark_master.py
```

Con eso, el script resuelve automaticamente:

- Entrada: `gs://tu-bucket/ml-100k`
- Salida: `gs://tu-bucket/processed`

Si necesitas rutas distintas, define estas variables y tienen prioridad:

- `MOVIELENS_INPUT_BASE_PATH`
- `MOVIELENS_OUTPUT_BASE_PATH`