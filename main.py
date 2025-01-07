from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer

# Cargar los datasets
movies_api1 = pd.read_csv("Data/recomendacion_api.csv")
genres_data1 = pd.read_csv("Data/genres_api.csv")

# Asegurarse de que ambas columnas tengan el mismo tipo
movies_api1["id"] = movies_api1["id"].astype(str)
genres_data1["id_original"] = genres_data1["id_original"].astype(str)

    # Procesar géneros para asegurar que no se repitan
movies_with_genres1 = pd.merge(movies_api1, genres_data1, left_on="id", right_on="id_original", how="left")

movies_with_genres2 = movies_with_genres1.sample(n=5000, random_state=42)

movies_with_genres2 = movies_with_genres2.groupby("id_original").agg({
    "title": "first",
    "name": lambda x: " ".join(set(x.dropna())),
    "vote_average": "first",  # Mantén la puntuación
    "popularity": "first"  # Mantén la popularidad
}).reset_index()

# Combinar texto y géneros en una nueva columna 'combined_features'
movies_with_genres2["combined_features"] = (
    movies_with_genres2["title"].fillna("") + " " +
    movies_with_genres2["name"].fillna("")
)
# Vectorizar el texto combinado usando TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies_with_genres2["combined_features"])

# Normalizar las puntuaciones numéricas
scaler = MinMaxScaler()
movies_with_genres2[["vote_average", "popularity"]] = scaler.fit_transform(
    movies_with_genres2[["vote_average", "popularity"]]
)

#reduzco el tipo de datos
movies_with_genres2 = movies_with_genres2.astype({'vote_average': 'float16', 'popularity': 'float16'})


# Agregar las características numéricas a la matriz TF-IDF
numerical_features = movies_with_genres2[["vote_average", "popularity"]].values
final_features = np.hstack((tfidf_matrix.toarray(), numerical_features))



# Calcular la matriz de similitud usando Cosine Similarity
similarity_matrix = cosine_similarity(final_features)


app = FastAPI()

# Función de recomendación
@app.get("/recomendacion/{titulo}")
def recomendacion1(titulo: str):
    try:
        # Verificar que el título esté en el dataset
        if titulo not in movies_with_genres2["title"].values:
            return {"mensaje": f"La película '{titulo}' no se encuentra en el dataset."}

        # Crear una nueva columna combinando título y géneros
        movies_with_genres2["combined_features"] = (
            movies_with_genres2["title"] + " " + movies_with_genres2["name"]
        )

        # Preprocesar las características combinadas
        count_vectorizer = CountVectorizer(stop_words="english")
        feature_matrix = count_vectorizer.fit_transform(movies_with_genres2["combined_features"])

        # Calcular la similitud coseno
        similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

        # Obtener el índice de la película dada
        idx = movies_with_genres2[movies_with_genres2["title"] == titulo].index[0]

        # Ordenar películas por similitud
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Obtener las 5 películas más similares (excluyendo duplicados y la misma película)
        seen_titles = set()  # Conjunto para rastrear títulos únicos
        seen_titles.add(titulo)
        top_movies = []
        for i, score in similarity_scores[1:]:  # Excluir la película misma
            movie_title = movies_with_genres2.iloc[i]["title"]
            if movie_title not in seen_titles:
                seen_titles.add(movie_title)
                top_movies.append(movie_title)
            if len(top_movies) == 5:  # Detener al alcanzar 5 recomendaciones
                break

        return {"recomendaciones": top_movies}
    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}
