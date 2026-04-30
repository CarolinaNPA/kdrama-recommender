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
    A[Inicio: Backtest SS26<br/>4 meses disponibles] --> B[Definir cutoff<br/>SS26_START_DATE]

    B --> C1[BigQuery: Mid<br/>identificar MIDs de SS26]
    B --> C2[BigQuery: Material_similarity<br/>obtener similares por MID]
    B --> C3[BigQuery: ne_weekly_sell_out<br/>histórico mensual de similares<br/>Date menor a cutoff]

    C1 --> D[Lista de MIDs<br/>a pronosticar]
    C2 --> D

    C3 --> E[Construir dataset<br/>mensual histórico]
    E --> E1[Mapear cada mes calendario<br/>a mes relativo de temporada<br/>mes 1, mes 2, mes 3, mes 4]

    D --> F[Para cada MID a pronosticar]
    E1 --> F

    F --> G{Tiene similares<br/>con datos?}
    G -->|No| H[Forecast igual a 0<br/>o fallback]
    G -->|Sí| I[Predicción mensual<br/>promedio de similares<br/>por mes relativo]

    I --> J[Forecast por MID:<br/>4 valores mensuales]
    H --> J

    J --> K[(forecast_propio.csv)]

    L[BigQuery: ne_weekly_sell_out<br/>venta real SS26<br/>Date entre cutoff y cutoff+4m] --> M[Real mensual<br/>por MID]

    K --> N[Comparación]
    M --> N
    O[(retail.csv / non_retail.csv<br/>output ARIMA producción)] --> N

    N --> P1[Métricas:<br/>MAE, MAPE, RMSE, Bias]
    N --> P2[Gráfica agregada:<br/>Real vs ARIMA vs Modelo propio]
    N --> P3[Top MIDs por error]
    N --> P4[Distribución del error]

    P1 --> Q[Reporte de comparación]
    P2 --> Q
    P3 --> Q
    P4 --> Q
```

