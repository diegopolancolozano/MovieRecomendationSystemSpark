"use client";

import { useEffect, useState } from "react";

type Recommendation = {
  movieId: number;
  title: string;
  genres: string | string[];
  rank?: number;
  ranking_score?: number;
  affinity_score?: number;
  cluster_avg_rating?: number;
  cluster_num_ratings?: number;
  score?: number;
  avg_rating?: number;
  ratings_count?: number;
  movie_rating?: { avg_rating: number; ratings_count: number };
  why?: {
    matched_user_genres?: string[];
    matched_cluster_genres?: string[];
    top_user_genres?: string[];
    top_cluster_genres?: string[];
  };
};

type UserResponse = {
  mode: "user";
  userId: number;
  k: number;
  cluster: { id: number; user_count: number };
  profile: {
    user_taste_matrix: Array<{ genre: string; weight: number }>;
    cluster_taste_matrix: Array<{ genre: string; weight: number }>;
  };
  recommendations: Recommendation[];
};

type SeedResponse = {
  mode: "seed";
  seed_movies: Array<{ movieId: number; title: string; genres: string[] }>;
  recommendations: Recommendation[];
};

type MovieSearchItem = {
  movieId: number;
  title: string;
  genres: string[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function formatGenres(genres: string | string[]) {
  return Array.isArray(genres) ? genres : genres.split("|").filter(Boolean);
}

export default function Page() {
  const [view, setView] = useState<"menu" | "user-input" | "seed-input" | "user" | "seed">("menu");
  const [userId, setUserId] = useState("12");
  const [selectedMovies, setSelectedMovies] = useState<number[]>([]);
  const [movieSearch, setMovieSearch] = useState("");
  const [movieResults, setMovieResults] = useState<MovieSearchItem[]>([]);
  const [movieSearchLoading, setMovieSearchLoading] = useState(false);
  const [movieSearchError, setMovieSearchError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [userResult, setUserResult] = useState<UserResponse | null>(null);
  const [seedResult, setSeedResult] = useState<SeedResponse | null>(null);

  useEffect(() => {
    if (view !== "seed-input") return;
    const query = movieSearch.trim();
    if (query.length < 2) {
      setMovieResults([]);
      setMovieSearchError("");
      setMovieSearchLoading(false);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      setMovieSearchLoading(true);
      setMovieSearchError("");
      try {
        const response = await fetch(
          `${API_BASE}/movies?query=${encodeURIComponent(query)}&limit=30`,
          { signal: controller.signal }
        );
        const data = (await response.json()) as { movies?: MovieSearchItem[]; error?: string };
        if (!response.ok) throw new Error(data.error || "No se pudo buscar películas");
        setMovieResults(data.movies ?? []);
      } catch (ex) {
        if (ex instanceof Error && ex.name === "AbortError") return;
        setMovieSearchError(ex instanceof Error ? ex.message : "Error inesperado");
        setMovieResults([]);
      } finally {
        setMovieSearchLoading(false);
      }
    }, 250);

    return () => {
      controller.abort();
      clearTimeout(timeout);
    };
  }, [movieSearch, view]);

  async function fetchUserRecommendations() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId: Number(userId), top_n: 10 }),
      });
      const data = (await response.json()) as UserResponse & { error?: string };
      if (!response.ok) throw new Error(data.error || "Error");
      setUserResult(data);
      setView("user");
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : "Error inesperado");
    } finally {
      setLoading(false);
    }
  }

  async function fetchSeedRecommendations() {
    if (!selectedMovies.length) {
      setError("Selecciona al menos una película");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const ratings = Object.fromEntries(selectedMovies.map((id) => [String(id), 5]));
      const response = await fetch(`${API_BASE}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ movieIds: selectedMovies, ratings, top_n: 10 }),
      });
      const data = (await response.json()) as SeedResponse & { error?: string };
      if (!response.ok) throw new Error(data.error || "Error");
      setSeedResult(data);
      setView("seed");
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : "Error inesperado");
    } finally {
      setLoading(false);
    }
  }

  // Menu principal
  if (view === "menu") {
    return (
      <main className="container menu">
        <div className="hero">
          <p className="eyebrow">MovieLens</p>
          <h1>¿Qué quieres ver?</h1>
          <p className="lead">Elige una opción para obtener recomendaciones personalizadas.</p>
        </div>

        <div className="grid-2">
          <button className="card-btn" onClick={() => { setView("user-input"); setError(""); }}>
            <h2>Por User ID</h2>
            <p>Dame tu ID de usuario y te muestro 10 películas que probablemente te gusten.</p>
          </button>

          <button className="card-btn" onClick={() => { setView("seed-input"); setError(""); }}>
            <h2>Por películas que me gustaron</h2>
            <p>Selecciona películas que disfrutaste y obtén recomendaciones basadas en tus gustos.</p>
          </button>
        </div>

        {error && <p className="error-banner">{error}</p>}
      </main>
    );
  }

  // Entrada de User ID
  if (view === "user-input") {
    return (
      <main className="container">
        <div className="panel">
          <button className="back-btn" onClick={() => setView("menu")}>← Atrás</button>
          <h2>Tu ID de usuario</h2>
          <input
            type="number"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Ej. 12"
            className="input"
          />
          <button className="primary-btn" onClick={fetchUserRecommendations} disabled={loading}>
            {loading ? "Cargando..." : "Recomendar"}
          </button>
          {error && <p className="error">{error}</p>}
        </div>
      </main>
    );
  }

  // Entrada de películas
  if (view === "seed-input") {
    const results = movieResults;

    return (
      <main className="container">
        <div className="panel">
          <button className="back-btn" onClick={() => setView("menu")}>← Atrás</button>
          <h2>Selecciona películas que te gustaron</h2>
          
          <input
            type="text"
            placeholder="Buscar películas..."
            value={movieSearch}
            onChange={(e) => setMovieSearch(e.target.value)}
            className="input"
          />

          {movieSearchError && <p className="error">{movieSearchError}</p>}
          {!movieSearchError && movieSearch.trim().length < 2 && (
            <p className="hint">Escribe al menos 2 letras para buscar.</p>
          )}
          {!movieSearchLoading && movieSearch.trim().length >= 2 && results.length === 0 && !movieSearchError && (
            <p className="hint">No encontramos coincidencias.</p>
          )}
          {movieSearchLoading && <p className="hint">Buscando...</p>}

          <div className="movie-list">
            {results.map((movie) => (
              <button
                key={movie.movieId}
                className={`movie-tag ${selectedMovies.includes(movie.movieId) ? "active" : ""}`}
                onClick={() =>
                  setSelectedMovies((prev) =>
                    prev.includes(movie.movieId)
                      ? prev.filter((id) => id !== movie.movieId)
                      : [...prev, movie.movieId]
                  )
                }
              >
                {movie.title}
              </button>
            ))}
          </div>

          <p className="hint">{selectedMovies.length} películas seleccionadas</p>

          <button className="primary-btn" onClick={fetchSeedRecommendations} disabled={loading || !selectedMovies.length}>
            {loading ? "Cargando..." : "Recomendar"}
          </button>
          {error && <p className="error">{error}</p>}
        </div>
      </main>
    );
  }

  // Resultados por usuario
  if (view === "user" && userResult) {
    return (
      <main className="container results">
        <button className="back-btn" onClick={() => setView("menu")}>← Menú</button>

        <div className="header">
          <h1>Recomendaciones para Usuario #{userResult.userId}</h1>
          <p className="subtitle">Cluster #{userResult.cluster.id} · {userResult.cluster.user_count} usuarios similares</p>
        </div>

        <div className="recommendations">
          {userResult.recommendations.map((item, idx) => (
            <article key={item.movieId} className="rec-card">
              <div className="rec-rank">#{idx + 1}</div>
              <div className="rec-content">
                <h3>{item.title}</h3>
                <p className="genres">{formatGenres(item.genres).join(" • ")}</p>

                <div className="rec-stats">
                  <span>Rating promedio: {(item.cluster_avg_rating ?? item.avg_rating ?? 0).toFixed(2)}/5</span>
                  <span>Votos: {(item.cluster_num_ratings ?? item.ratings_count ?? 0).toLocaleString()}</span>
                </div>

                {item.why && (
                  <div className="rec-why">
                    <p className="why-label">Por qué te la recomendamos:</p>
                    <p>
                      {item.why.matched_cluster_genres && item.why.matched_cluster_genres.length > 0
                        ? `Tiene géneros que te gustan (${item.why.matched_cluster_genres.join(", ")})`
                        : "Muy bien puntuada en tu cluster"}
                    </p>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </main>
    );
  }

  // Resultados por películas
  if (view === "seed" && seedResult) {
    return (
      <main className="container results">
        <button className="back-btn" onClick={() => setView("menu")}>← Menú</button>

        <div className="header">
          <h1>Te recomendamos basado en tu gusto</h1>
          <p className="subtitle">{seedResult.seed_movies.length} películas que marcaste como favoritas</p>
        </div>

        <div className="seed-movies">
          {seedResult.seed_movies.map((movie) => (
            <span key={movie.movieId} className="seed-tag">
              {movie.title}
            </span>
          ))}
        </div>

        <div className="recommendations">
          {seedResult.recommendations.map((item, idx) => (
            <article key={item.movieId} className="rec-card">
              <div className="rec-rank">#{idx + 1}</div>
              <div className="rec-content">
                <h3>{item.title}</h3>
                <p className="genres">{formatGenres(item.genres).join(" • ")}</p>

                <div className="rec-stats">
                  <span>Rating promedio: {(item.avg_rating ?? 0).toFixed(2)}/5</span>
                  <span>Votos: {(item.ratings_count ?? 0).toLocaleString()}</span>
                </div>

                {item.why && (
                  <div className="rec-why">
                    <p className="why-label">Por qué te la recomendamos:</p>
                    <p>
                      {item.why.matched_user_genres && item.why.matched_user_genres.length > 0
                        ? `Tiene géneros que te gustan (${item.why.matched_user_genres.join(", ")})`
                        : "Excelente calidad y popularidad"}
                    </p>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </main>
    );
  }

  return null;
}