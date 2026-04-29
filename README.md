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
    A[GCS:<br/>Mids_To_Proccess.parquet] --> D
    B[BigQuery: ne_weekly_sell_out<br/>ventas históricas] --> C[Agregación de sell-out<br/>por Similar_mid]
    B2[BigQuery: ne_master_customer<br/>filtros RETAIL / NON-RETAIL] --> C
    C --> D[Filtrado de MIDs<br/>a pronosticar]
    E[BigQuery: Material_similarity<br/>productos similares por MID] --> D
    D --> F{¿Self-reference<br/>únicamente?}
    F -->|Sí| G[Forecast = 0]
    F -->|No| H[eval_arima_model<br/>por MID]
    H --> H1{¿N de datos > 10?}
    H1 -->|Sí| H2[auto_arima]
    H1 -->|No| H3[Promedio ponderado]
    H2 --> I[Ajustes:<br/>penetración + cap 1.5x]
    H3 --> I
    I --> J[Concatenación con<br/>productos self-reference]
    G --> J
    J --> K1[(retail.csv)]
    J --> K2[(non_retail.csv)]
```



