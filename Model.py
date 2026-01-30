import pandas as pd
import numpy as np
import pickle



df = pd.read_csv('C:/Users/carod/Documents/RecomSyst/Kdramas_final.csv')

df["Main_vector_embedding_MM"] = df["Main_vector_embedding_MM"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)

df["Actors_vector_embedding_MM"] = df["Actors_vector_embedding_MM"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)

# **Modelo**
from sklearn.neighbors import NearestNeighbors

X = np.vstack(df["Main_vector_embedding_MM"].values)

model = NearestNeighbors(
    n_neighbors=6,
    metric="cosine"
)
model.fit(X)

Y = np.vstack(df["Actors_vector_embedding_MM"].values)

model2 = NearestNeighbors(
    n_neighbors=6,
    metric="cosine"
)
model2.fit(Y)

# **Main function**

def recommend_kdramas(title, df, model,model2, top_n=5):
    try:
        idx = df[df["Title"] == title].index[0]
    except IndexError:
        return f"Título '{title}' no encontrado."
    
    distances, indices = model.kneighbors([X[idx]])
    distances
    distances2, indices2 = model2.kneighbors([Y[idx]])
    distances2

    recs = df.iloc[indices[0]].copy()
    recs = recs[recs["Title"] != title]

    recs["final_score"] = (
        (1 - distances[0][1:]) * 0.7 +
        (1 - distances2[0][1:]) * 0.2 +
        recs["Rating_MM"] * 0.3 +
        (1 - abs(recs["Number of Episodes_MM"] - df.loc[idx, "Number of Episodes_MM"]) / df["Number of Episodes_MM"].max()) * 0.1 +
        (1 - abs(recs["Year of release_MM"] - df.loc[idx, "Year of release_MM"]) / 5) * 0.1

    )

    recs["confidence"] = (
    recs["final_score"] / recs["final_score"].max()
)

    return recs.sort_values("final_score", ascending=False).head(top_n)

# results = recommend_kdramas("Itaewon Class", df, model,model2)
# print(results)
# print('Correct')

# save models 
pickle.dump(model, open('model1.pkl', 'wb'))
pickle.dump(model2, open('model2.pkl', 'wb'))
# dataframe
pickle.dump(df, open('df_kdramas.pkl', 'wb'))