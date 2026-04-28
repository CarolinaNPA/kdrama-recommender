# K-Drama Recommendation System

A content-based hybrid recommendation system for Korean dramas using
semantic embeddings, metadata, and similarity search with KNN.

## Features
- Semantic similarity using text embeddings
- Actor-based similarity
- Metadata re-ranking (rating, episodes, release year)
- Interactive Streamlit app
- Poster retrieval via TMDB API

## Methodology
The system combines:
- Cosine similarity on narrative embeddings
- Cosine similarity on cast embeddings
- Re-ranking using normalized metadata features

Final recommendations are generated using a weighted scoring strategy.

```mermaid
flowchart TD
    A[BigQuery: new-era-xdl<br/>datos semanales de sell_out, stock, master_material] --> B[Consulta SQL con CTEs<br/>sell_out, stock, combine, material_filtered]
    C[GCS: Master Stores<br/>Partner, Concession, Wholesales] -->|filtro PBK = SI| B
    B --> D[(cluster_weekly.csv<br/>fila por cliente x semana)]
    D --> E[Ingeniería de características]
    E --> E1[Features de ventas]
    E --> E2[Features de stock]
    E --> E3[Ratios derivados]
    E1 --> F[Preprocesamiento]
    E2 --> F
    E3 --> F
    F --> F1[Winsorización percentil 99]
    F1 --> F2[QuantileTransformer]
    F2 --> F3[Pesos 50/50<br/>Size + Behaviour]
    F3 --> G[KMeans k=3]
    G --> H[Etiquetado determinístico<br/>por Total_SellOut promedio]
    H --> I1[(clients_clustered.csv)]
    H --> I2[tsne_clusters.png]
```

