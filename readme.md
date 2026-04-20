# Laboratorio Semana 7 - Clustering con Spark

**Estudiantes:**
- Ricardo Andrés Chamorro Martínez - A00399846
- Diego Armando Polanco Lozano - A00399926

## Descripción

Implementación de clustering K-Means sobre el dataset MovieLens 100K usando Apache Spark en un entorno distribuido (GCP).

## Estructura del Proyecto

### Archivos Principales
- `spark-kmeans.py` - **ÚNICO archivo necesario** - Script completo con K-Means

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

```
