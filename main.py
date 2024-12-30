from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np

# Crear la aplicación FastAPI
app = FastAPI()

# Definir modelos para la API
class RecommendationRequest(BaseModel):
    title: str
    n: int = 5

class ConfigurableNN:
    def __init__(self, metric="cosine", algorithm="auto"):
        self.nn = NearestNeighbors(metric=metric, algorithm=algorithm)

    def fit(self, features):
        self.nn.fit(features)

    def kneighbors(self, feature, n_neighbors=5):
        return self.nn.kneighbors(feature, n_neighbors=n_neighbors)

# Cargar y procesar los datasets
movies_api = pd.read_csv("Dataset/recomendacion_api.csv")
genres_data = pd.read_csv("Dataset/genres_api.csv")

movies_api["id"] = movies_api["id"].astype(str)
genres_data["id_original"] = genres_data["id_original"].astype(str)

movies_with_genres = pd.merge(movies_api, genres_data, left_on="id", right_on="id_original", how="left")

movies_with_genres = movies_with_genres.groupby("id_original").agg({
    "title": "first",
    "name": lambda x: " ".join(set(x.dropna())),
    "vote_average": "first",
    "popularity": "first"
}).reset_index()

movies_with_genres["combined_features"] = (
    movies_with_genres["title"].fillna("") + " " +
    movies_with_genres["name"].fillna("")
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies_with_genres["combined_features"])

scaler = MinMaxScaler()
movies_with_genres[["vote_average", "popularity"]] = scaler.fit_transform(
    movies_with_genres[["vote_average", "popularity"]]
)

numerical_features = csr_matrix(movies_with_genres[["vote_average", "popularity"]].values)

final_features = hstack([tfidf_matrix, numerical_features])

# Instancia configurable de NearestNeighbors
nn = ConfigurableNN(metric="cosine", algorithm="auto")
nn.fit(final_features)

# Función para encontrar películas similares
def find_similar_movies(title, n=5):
    if title not in movies_with_genres["title"].values:
        raise ValueError(f"La película '{title}' no se encontró en la base de datos.")

    idx = movies_with_genres[movies_with_genres["title"] == title].index[0]
    distances, indices = nn.kneighbors(final_features[idx], n_neighbors=n+1)
    recommended_titles = movies_with_genres.iloc[indices[0]]['title'].tolist()
    recommended_titles = [t for t in recommended_titles if t != title]

    # Rellenar con títulos adicionales si faltan recomendaciones
    if len(recommended_titles) < n:
        extra_titles = movies_with_genres[~movies_with_genres["title"].isin(recommended_titles + [title])]["title"].tolist()
        recommended_titles.extend(extra_titles[:n - len(recommended_titles)])

    return recommended_titles[:n]

# Endpoints de la API
@app.get("/")
def root():
    return {"message": "Bienvenido a la API de recomendación de películas."}

@app.post("/recommendations/")
def get_recommendations(request: RecommendationRequest):
    try:
        if not request.title:
            raise HTTPException(status_code=400, detail="El título de la película es obligatorio.")
        recommendations = find_similar_movies(request.title, request.n)
        return {"title": request.title, "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")