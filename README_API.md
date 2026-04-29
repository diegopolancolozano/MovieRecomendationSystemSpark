# README — Cómo ejecutar la API de recomendaciones

Este archivo explica los pasos mínimos para ejecutar el servicio HTTP que expone
las recomendaciones precalculadas.

Requisitos mínimos
- Python 3.8+ (probado con 3.11)
- Dataset MovieLens en `ml-1m/` (archivos `movies.dat` y `ratings.dat`)

Dependencias
- Para ejecutar sólo `recommendation_api.py` NO necesitas paquetes externos.
- Si quieres ejecutar los scripts de Spark (`spark-kmeans*.py`) instala `pyspark`.
  Se incluye `requirements.txt` con `pyspark>=3.3.0` como opción.

Pasos (Windows PowerShell)

1) Crear y activar entorno virtual
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2) (Opcional) instalar dependencias
```powershell
pip install -r requirements.txt
```

3) Ejecutar la API
```powershell
python recommendation_api.py --host 127.0.0.1 --port 8000
```

Endpoints útiles
- `GET /recommendations` — devuelve todas las recomendaciones (JSON con `user_id`, `cluster`, `recommendations`)
- `GET /recommendations/{user_id}` — recomendaciones de un usuario (200 o 404 si no existe)

Comprobación rápida
```bash
curl -i "http://127.0.0.1:8000/recommendations"
curl -i "http://127.0.0.1:8000/recommendations/6023"
```
