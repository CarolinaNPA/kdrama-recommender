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
    A[(GCS:<br/>Mids_To_Proccess.parquet)] --> D
    
    B1[BigQuery: ne_weekly_sell_out<br/>query_pzs] --> C1[get_final_sellouts<br/>retail]
    B1 --> C2[get_final_sellouts<br/>non-retail]
    
    B2[BigQuery: ne_master_customer<br/>query_customers_retail] --> C1
    B3[BigQuery: ne_master_customer<br/>query_customers_non_retail] --> C2
    
    B4[BigQuery: Material_similarity<br/>query_similarity] --> D[Filtrado de MIDs<br/>a pronosticar]
    
    D --> E{MID con<br/>self-reference?}
    E -->|Sí| F[Forecast igual a 0<br/>asignación directa]
    E -->|No| G[eval_arima_model<br/>por MID]
    
    C1 --> G
    C2 --> G
    
    G --> G1{Cantidad de<br/>datos mayor a 10?}
    G1 -->|Sí| G2[auto_arima<br/>n_periods igual a 9]
    G1 -->|No| G3[Promedio ponderado<br/>pesos 0.5 a 1.0]
    
    G2 --> H[Ajustes:<br/>penetración de canal<br/>cap 1.5x histórico]
    G3 --> H
    
    H --> I[Concatenación con<br/>productos self-reference]
    F --> I
    
    I --> J1[(retail.csv)]
    I --> J2[(non_retail.csv)]
```
