import pickle
import streamlit as st
import requests
import numpy as np

st.title('k-Drama Recommendation System')
st.caption('Recommended due to thematic similarity, tone, and narrative structure')

#APIKEY
import os

API_KEY = os.getenv("TMDB_API_KEY")

#function to get poster
@st.cache_data
def get_poster_url(title):
    url = f'https://api.themoviedb.org/3/search/tv?api_key={API_KEY}&query={title}&language=en-US'
    response = requests.get(url).json()
    if response['results']:
        poster_path = response['results'][0]['poster_path']
        return f'https://image.tmdb.org/t/p/w500{poster_path}' if poster_path else None
    return None


# kdrama system
kdrama = pickle.load(open('df_kdramas.pkl', 'rb'))

#model embedding

model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
X = np.vstack(kdrama["Main_vector_embedding_MM"].values)
Y = np.vstack(kdrama["Actors_vector_embedding_MM"].values)

kdrama_list = kdrama['Title'].tolist()

# write Kdrama
selected_kdrama = st.selectbox(
    '**What was your last K-Drama?**', kdrama_list
)

# Show reccomendation
# function 
def recommend_kdramas(title,X,Y):
    try:
        idx = kdrama[kdrama["Title"] == title].index[0]
    except IndexError:
        return f"Título '{title}' no encontrado."
    
    distances, indices = model1.kneighbors([X[idx]])
    
    distances2, indices2 = model2.kneighbors([Y[idx]])
    
    neighbor_idx = indices[0][1:]
    sim_main = 1 - distances[0][1:]
    sim_cast = 1 - distances2[0][1:]

    recs = kdrama.iloc[neighbor_idx].copy()
    
    # recs = kdrama.iloc[indices[0]].copy()
    recs = recs[recs["Title"] != title]

    recs["final_score"] = (
        0.5 * sim_main +
        0.1 * sim_cast +
        0.2 * recs["Rating_MM"] +
        0.1 * (1 - abs(recs["Number of Episodes_MM"] - kdrama.loc[idx, "Number of Episodes_MM"]) 
            / kdrama["Number of Episodes_MM"].max()) +
        0.1 * (1 - abs(recs["Year of release_MM"] - kdrama.loc[idx, "Year of release_MM"]) / 5)
    )

    recs["confidence"] = (
    recs["final_score"] / recs["final_score"].max()
)

    return recs.sort_values("final_score", ascending=False).head(5)

#show recommendation
if st.button('Show Recommendation'):

    col1,col2 = st.columns([1,2])

    with col1:
    #poster
        poster_url = get_poster_url(selected_kdrama)
        if poster_url:
            st.image(poster_url, width=150, caption=selected_kdrama)  
        else:
            st.write("Poster not found")
    with col2:
        title_selected = kdrama[kdrama['Title'] == selected_kdrama]

        for i, row in title_selected.iterrows():
            title_desc = row['Description']
            title_genre = row['Genre']
            title_year = row['Year of release']
            title_rating = row['Rating']

        st.write(f"**Description:** {title_desc}")
        st.write(f"**Genre:** {title_genre}")
        st.write(f"**Year of release:** {title_year}")
        st.write(f"**Rate:** {title_rating}")

    st.title(f'**Recommend K-Drama based on**: {selected_kdrama}')
    recommendations = recommend_kdramas(selected_kdrama,X,Y)
    if isinstance(recommendations, str):
        st.write(f'This are the recommendations')  
    else:
        for i, row in recommendations.iterrows():
            title = row['Title']
            description = row['Description']
            genre = row['Genre']
            rating = row['Rating']
            confidence = row['confidence']
            year = row['Year of release']

            col1, col2 = st.columns([1, 2])

            #poster
            with col1:
                poster_url = get_poster_url(title)
                if poster_url:
                    st.image(poster_url, width=150, caption=title)  
                else:
                    st.write("Poster not found")  
            with col2:
                st.write(f"**{title}**")
                st.write(f"**Description:** {description}")
                st.write(f"**Genre:** {genre}")
                st.write(f'**Year of release:** {year}')
                st.write(f"**Rating:** {rating}")
                st.write(f"**Confidence:** {confidence:.2f}")
            st.write("---")