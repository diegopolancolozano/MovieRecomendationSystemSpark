# Laboratorio Semana 7 - Clustering con Spark

**Estudiantes:**
- Ricardo Andrés Chamorro Martínez - A00399846
- Diego Armando Polanco Lozano - A00399926

## Descripción

Implementación de clustering K-Means sobre el dataset MovieLens 100K usando Apache Spark en un entorno distribuido (GCP).

## Estructura del Proyecto

### Archivos Principales
- `spark-kmeans.py` - **ÚNICO archivo necesario** - Script completo con K-Means
- `run_cluster.sh` - Script para ejecutar fácilmente en el cluster

### Documentación
- `INSTRUCCIONES_GCP.md` - Guía paso a paso para configurar el cluster
- `ARQUITECTURA_DISTRIBUIDA.md` - Explicación detallada de cómo funciona la distribución
- `DIAGRAMA_SIMPLE.md` - Diagrama visual simple de la arquitectura
- `GUIA_INFORME.md` - Estructura y contenido para el informe
- `CHECKLIST.md` - Lista de verificación completa
- `COMANDOS_UTILES.md` - Referencia rápida de comandos

### Archivos Opcionales (no necesarios)
- `spark_master.py` - Funciones auxiliares (no usadas por spark-kmeans.py)
- `spark_worker.py` - Funciones de features (no usadas por spark-kmeans.py)

## Características Implementadas

1. **Carga y exploración de datos**
   - Carga desde Google Cloud Storage
   - Análisis de usuarios, películas y distribución de ratings

2. **Construcción de features**
   - Extracción de géneros de películas
   - Weighted ratings por género
   - Normalización de features

3. **Preparación de datos**
   - VectorAssembler para ensamblar features
   - StandardScaler para normalización

4. **K-Means Clustering**
   - Prueba con K=3, 5, 8
   - Evaluación con Silhouette Score
   - Selección automática del mejor K

5. **Análisis de resultados**
   - Distribución de usuarios por cluster
   - Top géneros preferidos por cluster
   - Caracterización de cada grupo

## Ejecución en GCP

### ¿Cómo se ejecuta?

**TÚ ejecutas manualmente** en la VM Master:

```bash
# Opción 1: Directamente
spark-submit spark-kmeans.py

# Opción 2: Con el script auxiliar
./run_cluster.sh
```

**NO hay ningún archivo que llame automáticamente a spark-kmeans.py.**

Ver `FLUJO_EJECUCION.md` para entender el flujo completo.

### Configuración del Cluster

Ver instrucciones detalladas en `INSTRUCCIONES_GCP.md`

### Ejecución Rápida

1. Configurar cluster según `INSTRUCCIONES_GCP.md`
2. Editar `run_cluster.sh` con tus IPs y bucket
3. Ejecutar:
   ```bash
   chmod +x run_cluster.sh
   ./run_cluster.sh
   ```

### Ejecución Manual

```bash
# Configurar variables
export MOVIELENS_GCS_BUCKET="gs://tu-bucket-movielens"
export SPARK_MASTER_URL="spark://IP_MASTER:7077"

# Ejecutar
spark-submit \
  --master spark://IP_MASTER:7077 \
  --executor-memory 4g \
  --executor-cores 2 \
  --num-executors 2 \
  spark-kmeans.py
```

## Resultados

Los resultados se guardan en GCS:
- `gs://tu-bucket/output/clusters_kX/` - Asignación de usuarios a clusters

La consola mostrará:
- Análisis exploratorio del dataset
- Silhouette Score para cada K
- Distribución de usuarios por cluster
- Top 5 géneros preferidos por cada cluster
