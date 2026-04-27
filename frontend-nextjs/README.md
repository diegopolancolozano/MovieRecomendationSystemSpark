# Frontend Next.js

Frontend sencillo para obtener recomendaciones de películas.

## Características

- **Por usuario**: ingresa tu ID de usuario y obtén 10 películas recomendadas.
- **Por películas favoritas**: selecciona películas que te gustaron y recibe recomendaciones basadas en tus gustos.

Cada película muestra:
- Título y géneros
- Rating promedio y cantidad de votos
- Explicación clara de por qué te la recomendamos

## Setup

1. Instala dependencias en esta carpeta:

```bash
npm install
```

2. Levanta el backend en la raíz (en otra terminal):

```bash
python recommendation_api.py
```

3. Ejecuta el frontend:

```bash
npm run dev
```

Abre http://127.0.0.1:3000 en tu navegador.

## Configuración

Si el backend corre en otra URL, define la variable de entorno:

```bash
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

## Build

```bash
npm run build
npm run start
```