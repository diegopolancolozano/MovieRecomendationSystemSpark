"""Explainable HTTP API and frontend for the MovieLens recommendation system.

Endpoints:
    GET  /                          -> frontend HTML
    GET  /health                    -> service health and metadata
    GET  /movies?query=...          -> search movies by title
    GET  /recommendations           -> all users with their recommendations (Lab 10)
  GET  /recommendations/{user_id} -> single user recommendations (Lab 10)
  POST /recommendations           -> recommendation payload (legacy)

The POST body accepts either:
  - {"userId": 1, "k": 10, "top_n": 10}
  - {"movieIds": [1, 50, 260], "ratings": {"1": 5, "50": 4}, "top_n": 10}

For userId, the API serves the precomputed cluster-based recommendations and
returns an explanation with:
  - the user's cluster
  - the user's taste matrix
  - the cluster taste matrix
  - the movie rating stats used to rank recommendations
  - the raw recommendation score breakdown

For movieIds, it builds a lightweight content-based profile from the marked
movies and scores the catalog using genre affinity, rating quality and popularity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_DATA_DIR = ROOT_DIR / "ml-1m"
FRONTEND_FILE = ROOT_DIR / "index.html"
POSITIVE_THRESHOLD = 3.0


@dataclass(frozen=True)
class MovieRecord:
    movie_id: int
    title: str
    genres: List[str]


@dataclass(frozen=True)
class RatingRecord:
    user_id: int
    movie_id: int
    rating: float


class RecommendationService:
    def __init__(self, data_dir: Path, output_dir: Path) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.movies = self._load_movies()
        self.user_ratings, self.movie_rating_stats, self.global_rating_stats = self._load_ratings()
        self.genre_names = self._load_genre_names()
        self.best_k = self._load_best_k()
        self._cluster_assignments_cache: Dict[int, Dict[int, int]] = {}
        self._cluster_recommendations_cache: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        self._user_profile_cache: Dict[int, Dict[str, Any]] = {}
        self._cluster_profile_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._lab10_cache: Optional[List[Dict[str, Any]]] = None
        print(
            f"[INFO] Loaded {len(self.movies)} movies, {len(self.user_ratings)} users, best_k={self.best_k}"
        )

    def _load_movies(self) -> Dict[int, MovieRecord]:
        movies_path = self.data_dir / "movies.dat"
        if not movies_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {movies_path}")

        movies: Dict[int, MovieRecord] = {}
        with movies_path.open("r", encoding="latin-1") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parts = raw_line.split("::")
                if len(parts) < 3:
                    continue
                movie_id = int(parts[0])
                title = parts[1]
                genres = [
                    genre
                    for genre in parts[2].split("|")
                    if genre and genre != "(no genres listed)"
                ]
                movies[movie_id] = MovieRecord(movie_id=movie_id, title=title, genres=genres)
        return movies

    def _load_genre_names(self) -> List[str]:
        genre_names = []
        seen = set()
        for movie in self.movies.values():
            for genre in movie.genres:
                if genre not in seen:
                    seen.add(genre)
                    genre_names.append(genre)
        return sorted(genre_names)

    def search_movies(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned:
            return []

        normalized = cleaned.casefold()
        limit = max(1, min(limit, 50))
        matches: List[tuple[int, int, int, str, int, MovieRecord]] = []

        for movie in self.movies.values():
            title_norm = movie.title.casefold()
            index = title_norm.find(normalized)
            if index == -1:
                continue
            starts_with = 1 if index == 0 else 0
            matches.append((starts_with, index, len(movie.title), movie.title, movie.movie_id, movie))

        matches.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))

        results: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()

        if normalized.isdigit():
            movie_id = int(normalized)
            movie = self.movies.get(movie_id)
            if movie is not None:
                results.append({"movieId": movie.movie_id, "title": movie.title, "genres": movie.genres})
                seen_ids.add(movie.movie_id)

        for _, _, _, _, movie_id, movie in matches:
            if movie_id in seen_ids:
                continue
            results.append({"movieId": movie.movie_id, "title": movie.title, "genres": movie.genres})
            if len(results) >= limit:
                break

        return results

    def _load_ratings(self) -> tuple[Dict[int, List[RatingRecord]], Dict[int, Dict[str, float]], Dict[str, float]]:
        ratings_path = self.data_dir / "ratings.dat"
        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {ratings_path}")

        user_ratings: DefaultDict[int, List[RatingRecord]] = defaultdict(list)
        movie_totals: DefaultDict[int, float] = defaultdict(float)
        movie_counts: DefaultDict[int, int] = defaultdict(int)
        global_total = 0.0
        global_count = 0

        with ratings_path.open("r", encoding="latin-1") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parts = raw_line.split("::")
                if len(parts) < 4:
                    continue
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = float(parts[2])
                record = RatingRecord(user_id=user_id, movie_id=movie_id, rating=rating)
                user_ratings[user_id].append(record)
                movie_totals[movie_id] += rating
                movie_counts[movie_id] += 1
                global_total += rating
                global_count += 1

        movie_stats: Dict[int, Dict[str, float]] = {}
        for movie_id, count in movie_counts.items():
            movie_stats[movie_id] = {
                "avg_rating": movie_totals[movie_id] / count,
                "count": float(count),
            }

        global_stats = {
            "avg_rating": global_total / global_count if global_count else 0.0,
            "count": float(global_count),
        }
        return dict(user_ratings), movie_stats, global_stats

    def _load_best_k(self) -> int:
        metrics_path = self.output_dir / "evaluation_metrics.json"
        if not metrics_path.exists():
            return 10

        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 10

        best_k = payload.get("best_k_by_precision")
        if isinstance(best_k, int):
            return best_k
        return 10

    def _load_cluster_assignments(self, k_value: int) -> Dict[int, int]:
        if k_value in self._cluster_assignments_cache:
            return self._cluster_assignments_cache[k_value]

        path = self.output_dir / f"clusters_k{k_value}" / "data.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing cluster assignments file: {path}")

        assignments: Dict[int, int] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                user_id = int(row["userId"])
                cluster_id = int(row["cluster"])
                assignments[user_id] = cluster_id

        self._cluster_assignments_cache[k_value] = assignments
        return assignments

    def _load_cluster_recommendations(self, k_value: int) -> Dict[int, List[Dict[str, Any]]]:
        if k_value in self._cluster_recommendations_cache:
            return self._cluster_recommendations_cache[k_value]

        path = self.output_dir / f"recommendations_top10_k{k_value}" / "data.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing recommendation file: {path}")

        grouped: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                user_id = int(row["userId"])
                grouped[user_id].append(
                    {
                        "userId": user_id,
                        "cluster": int(row["cluster"]),
                        "movieId": int(row["movieId"]),
                        "title": row["title"],
                        "genres": row["genres"],
                        "ranking_score": float(row["ranking_score"]),
                        "affinity_score": float(row["affinity_score"]),
                        "cluster_avg_rating": float(row["cluster_avg_rating"]),
                        "cluster_num_ratings": int(float(row["cluster_num_ratings"])),
                        "rank": int(row["rank"]),
                    }
                )

        recommendations = {user_id: sorted(items, key=lambda item: item["rank"]) for user_id, items in grouped.items()}
        self._cluster_recommendations_cache[k_value] = recommendations
        return recommendations

    def _build_taste_profile(self, ratings: List[RatingRecord]) -> Dict[str, Any]:
        chosen_ratings = [record for record in ratings if record.rating > POSITIVE_THRESHOLD]
        if not chosen_ratings:
            chosen_ratings = ratings

        genre_weights: Counter[str] = Counter()
        seed_movies: List[Dict[str, Any]] = []

        for record in chosen_ratings:
            movie = self.movies.get(record.movie_id)
            if movie is None:
                continue
            genre_count = max(len(movie.genres), 1)
            weight = record.rating / genre_count
            seed_movies.append(
                {
                    "movieId": movie.movie_id,
                    "title": movie.title,
                    "genres": movie.genres,
                    "rating": round(record.rating, 2),
                    "weight": round(weight, 4),
                }
            )
            for genre in movie.genres:
                genre_weights[genre] += weight

        return {
            "seed_movies": sorted(seed_movies, key=lambda item: (-item["rating"], item["title"])),
            "taste_matrix": self._format_taste_matrix(genre_weights),
        }

    def _format_taste_matrix(self, genre_weights: Counter[str] | Dict[str, float]) -> List[Dict[str, Any]]:
        total_weight = float(sum(genre_weights.values()))
        matrix: List[Dict[str, Any]] = []
        for genre in self.genre_names:
            weight = float(genre_weights.get(genre, 0.0))
            share = weight / total_weight if total_weight else 0.0
            matrix.append(
                {
                    "genre": genre,
                    "weight": round(weight, 4),
                    "share": round(share, 4),
                }
            )
        matrix.sort(key=lambda item: (-item["weight"], item["genre"]))
        return matrix

    def _build_user_profile(self, user_id: int) -> Dict[str, Any]:
        if user_id in self._user_profile_cache:
            return self._user_profile_cache[user_id]

        ratings = self.user_ratings.get(user_id)
        if not ratings:
            raise KeyError(f"User {user_id} not found in ratings.dat")

        profile = self._build_taste_profile(ratings)
        rated_movie_ids = [record.movie_id for record in ratings if record.rating > POSITIVE_THRESHOLD]
        profile["userId"] = user_id
        profile["ratings_count"] = len(ratings)
        profile["positive_ratings_count"] = len(rated_movie_ids)
        profile["rated_movies"] = profile["seed_movies"]
        self._user_profile_cache[user_id] = profile
        return profile

    def _build_cluster_profile(self, k_value: int, cluster_id: int) -> Dict[str, Any]:
        cache_by_k = self._cluster_profile_cache.setdefault(k_value, {})
        if cluster_id in cache_by_k:
            return cache_by_k[cluster_id]

        assignments = self._load_cluster_assignments(k_value)
        cluster_users = [user_id for user_id, assigned_cluster in assignments.items() if assigned_cluster == cluster_id]
        if not cluster_users:
            raise KeyError(f"Cluster {cluster_id} not found for K={k_value}")

        accumulated_weights: DefaultDict[str, float] = defaultdict(float)
        sample_profiles = 0
        for user_id in cluster_users:
            try:
                profile = self._build_user_profile(user_id)
            except KeyError:
                continue
            sample_profiles += 1
            for row in profile["taste_matrix"]:
                accumulated_weights[row["genre"]] += float(row["share"])

        if sample_profiles == 0:
            raise KeyError(f"Unable to build cluster profile for cluster {cluster_id} and K={k_value}")

        averaged_weights = {genre: weight / sample_profiles for genre, weight in accumulated_weights.items()}
        profile = {
            "cluster": cluster_id,
            "k": k_value,
            "user_count": len(cluster_users),
            "sampled_users": sample_profiles,
            "taste_matrix": self._format_taste_matrix(averaged_weights),
        }
        cache_by_k[cluster_id] = profile
        return profile

    def _enrich_recommendation(
        self,
        recommendation: Dict[str, Any],
        user_profile: Dict[str, Any],
        cluster_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        user_top_genres = [row["genre"] for row in user_profile["taste_matrix"][:5]]
        cluster_top_genres = [row["genre"] for row in cluster_profile["taste_matrix"][:5]]
        movie_genres = [genre for genre in recommendation["genres"].split("|") if genre]
        matched_user = [genre for genre in movie_genres if genre in user_top_genres]
        matched_cluster = [genre for genre in movie_genres if genre in cluster_top_genres]
        movie_stats = self.movie_rating_stats.get(
            recommendation["movieId"],
            self.global_rating_stats,
        )

        enriched = dict(recommendation)
        enriched["movie_rating"] = {
            "avg_rating": round(float(movie_stats["avg_rating"]), 4),
            "ratings_count": int(movie_stats["count"]),
        }
        enriched["why"] = {
            "matched_user_genres": matched_user,
            "matched_cluster_genres": matched_cluster,
            "top_user_genres": user_top_genres,
            "top_cluster_genres": cluster_top_genres,
        }
        return enriched

    def get_user_recommendations(self, user_id: int, top_n: int = 10, k_value: Optional[int] = None) -> Dict[str, Any]:
        chosen_k = k_value or self.best_k
        assignments = self._load_cluster_assignments(chosen_k)
        if user_id not in assignments:
            raise KeyError(f"User {user_id} not found in clusters for K={chosen_k}")

        cluster_id = assignments[user_id]
        recommendations_by_user = self._load_cluster_recommendations(chosen_k)
        if user_id not in recommendations_by_user:
            raise KeyError(f"User {user_id} has no precomputed recommendations for K={chosen_k}")

        user_profile = self._build_user_profile(user_id)
        cluster_profile = self._build_cluster_profile(chosen_k, cluster_id)
        items = recommendations_by_user[user_id][:top_n]

        return {
            "mode": "user",
            "userId": user_id,
            "k": chosen_k,
            "cluster": {
                "id": cluster_id,
                "user_count": cluster_profile["user_count"],
                "sampled_users": cluster_profile["sampled_users"],
            },
            "profile": {
                "user_taste_matrix": user_profile["taste_matrix"],
                "cluster_taste_matrix": cluster_profile["taste_matrix"],
                "seed_movies": user_profile["seed_movies"],
                "ratings_count": user_profile["ratings_count"],
                "positive_ratings_count": user_profile["positive_ratings_count"],
            },
            "recommendations": [
                self._enrich_recommendation(item, user_profile, cluster_profile) for item in items
            ],
        }

    def get_seed_recommendations(
        self,
        movie_ids: List[int],
        top_n: int = 10,
        ratings: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        if not movie_ids:
            raise ValueError("movieIds cannot be empty")

        ratings = ratings or {}
        seed_movies: List[Dict[str, Any]] = []
        genre_weights: Counter[str] = Counter()
        marked_ids = set(movie_ids)

        for movie_id in movie_ids:
            movie = self.movies.get(movie_id)
            if movie is None:
                continue
            weight = float(ratings.get(movie_id, 1.0))
            seed_movies.append(
                {
                    "movieId": movie.movie_id,
                    "title": movie.title,
                    "genres": movie.genres,
                    "rating": round(weight, 2),
                    "weight": round(weight, 4),
                }
            )
            for genre in movie.genres:
                genre_weights[genre] += weight

        if not seed_movies:
            raise KeyError("None of the provided movieIds exist in the dataset")

        global_avg = float(self.global_rating_stats["avg_rating"])
        max_count = max((int(stats["count"]) for stats in self.movie_rating_stats.values()), default=1)
        total_seed_weight = sum(genre_weights.values()) or 1.0
        scored: List[Dict[str, Any]] = []

        for movie_id, movie in self.movies.items():
            if movie_id in marked_ids:
                continue

            affinity = sum(genre_weights.get(genre, 0.0) for genre in movie.genres)
            affinity_norm = affinity / total_seed_weight
            movie_stats = self.movie_rating_stats.get(movie_id, {"avg_rating": global_avg, "count": 0.0})
            avg_rating = float(movie_stats["avg_rating"])
            count = int(movie_stats["count"])
            rating_norm = avg_rating / 5.0
            popularity_norm = math.log1p(count) / math.log1p(max_count) if max_count > 1 else 0.0
            score = 0.60 * affinity_norm + 0.25 * rating_norm + 0.15 * popularity_norm
            scored.append(
                {
                    "movieId": movie_id,
                    "title": movie.title,
                    "genres": movie.genres,
                    "score": round(score, 4),
                    "affinity_norm": round(affinity_norm, 4),
                    "avg_rating": round(avg_rating, 4),
                    "ratings_count": count,
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["title"], item["movieId"]))
        return {
            "mode": "seed",
            "top_n": top_n,
            "seed_movies": seed_movies,
            "taste_matrix": self._format_taste_matrix(genre_weights),
            "recommendations": scored[:top_n],
        }


    def get_lab10_recommendations(self) -> List[Dict[str, Any]]:
        """Carga y cachea las recomendaciones desde el JSON generado por Spark.

        Formato de salida requerido por Lab 10:
          [{"user_id": 1, "cluster": 2, "recommendations": [{"movie_id": ..., "movie_title": ..., "score": ...}]}]
        """
        if self._lab10_cache is not None:
            return self._lab10_cache

        path = self.output_dir / f"recommendations_k{self.best_k}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {path}. Ejecuta spark-kmeans-local.py primero."
            )

        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        result: List[Dict[str, Any]] = []

        for user_id_str, items in raw.items():
            sorted_items = sorted(items, key=lambda x: x["rank"])
            cluster = sorted_items[0]["cluster"] if sorted_items else -1
            result.append(
                {
                    "user_id": int(user_id_str),
                    "cluster": cluster,
                    "recommendations": [
                        {
                            "movie_id": item["movieId"],
                            "movie_title": item["title"],
                            "score": round(float(item["cluster_avg_rating"]), 4),
                        }
                        for item in sorted_items
                    ],
                }
            )

        result.sort(key=lambda x: x["user_id"])
        self._lab10_cache = result
        return result


class RecommendationHandler(BaseHTTPRequestHandler):
    service: RecommendationService

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_text: str) -> None:
        body = html_text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query_params = parse_qs(parsed.query)
        import json as _json
        with open("C:\\Users\\DELL\\Desktop\\api_debug.log", "a") as f:
            _json.dump({"path": self.path, "route": route}, f)
            f.write("\n")
        if route in {"/", "/index.html"}:
            if FRONTEND_FILE.exists():
                self._send_html(FRONTEND_FILE.read_text(encoding="utf-8"))
            else:
                self._send_html(
                    "<html><body><h1>Recommendation API</h1><p>Frontend file missing.</p></body></html>"
                )
            return

        if route == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "best_k": self.service.best_k,
                    "movies_loaded": len(self.service.movies),
                    "users_loaded": len(self.service.user_ratings),
                },
            )
            return

        if route == "/movies":
            query = (query_params.get("query") or [""])[0]
            limit_raw = (query_params.get("limit") or ["20"])[0]
            try:
                limit = int(limit_raw)
            except ValueError:
                limit = 20
            data = self.service.search_movies(query, limit=limit)
            self._send_json(HTTPStatus.OK, {"movies": data})
            return

        # Lab 10 — GET /recommendations (todas las recomendaciones)
        if route == "/recommendations":
            try:
                data = self.service.get_lab10_recommendations()
                self._send_json(HTTPStatus.OK, data)
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        # Lab 10 — GET /recommendations/{user_id}
        match = re.fullmatch(r"/recommendations/(\d+)", route)
        if match:
            user_id = int(match.group(1))
            try:
                all_recs = self.service.get_lab10_recommendations()
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            user_data = next((u for u in all_recs if u["user_id"] == user_id), None)
            if user_data is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Usuario {user_id} no encontrado"})
            else:
                self._send_json(HTTPStatus.OK, user_data)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.rstrip("/")
        if route != "/recommendations":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
            return

        top_n = int(payload.get("top_n", 10))
        k_value = payload.get("k")

        try:
            if payload.get("userId") is not None:
                response = self.service.get_user_recommendations(
                    int(payload["userId"]),
                    top_n=top_n,
                    k_value=int(k_value) if k_value is not None else None,
                )
            elif payload.get("movieIds"):
                ratings_payload = payload.get("ratings") or {}
                ratings = {int(key): float(value) for key, value in ratings_payload.items()}
                response = self.service.get_seed_recommendations(
                    [int(movie_id) for movie_id in payload["movieIds"]],
                    top_n=top_n,
                    ratings=ratings,
                )
            else:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Send either userId or movieIds in the request body"},
                )
                return
        except (ValueError, KeyError, FileNotFoundError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        msg = f"[API] {self.address_string()} - {format % args}"
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens recommendation API")
    parser.add_argument("--host", default=os.getenv("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_PATH", str(DEFAULT_OUTPUT_DIR)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = RecommendationService(Path(args.data_dir), Path(args.output_dir))
    RecommendationHandler.service = service

    server = ThreadingHTTPServer((args.host, args.port), RecommendationHandler)
    print(f"[INFO] Recommendation API running on http://{args.host}:{args.port}")
    print(f"[INFO] Best K loaded from output: {service.best_k}")
    print("[INFO] Endpoints:")
    print("[INFO]   GET  /                          -> frontend HTML")
    print("[INFO]   GET  /health                    -> health check")
    print("[INFO]   GET  /movies?query=...          -> search movies by title")
    print("[INFO]   GET  /recommendations           -> all users (Lab 10)")
    print("[INFO]   GET  /recommendations/{user_id} -> single user (Lab 10)")
    print("[INFO]   POST /recommendations           -> recommendation payload")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[INFO] Shutting down API")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
