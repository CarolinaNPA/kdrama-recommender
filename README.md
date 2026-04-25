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
    A[sellout_size.csv] --> LD[load_source_data]
    B[similarity_size.csv] --> LD
    C[new_season_size.csv] --> LD
    LD --> NULL[drop_nulls_and_report]
    NULL --> PLC[normalize_months - PLC string to int]
    PLC --> PREP[Preprocessing - variant-specific - see Section 5]
    PREP --> TRN[df_train - monthly features with targets]
    PREP --> INF[df_infer - monthly features for new MIDs]
    D[clients_clustered.csv] --> CLU[enrich_with_cluster]
    TRN --> CLU
    INF --> CLU
    CLU --> FIL[Filter to CLUSTER_VALUE]
    FIL --> CAP[Cap target at CAP_Q p99]
    CAP --> LOG[log1p target transform]
    LOG --> TUNE[tune_extratrees or tune_extratrees_ts]
    TUNE --> PKG[Model package - model, medians, cap, feature_names]
    PKG --> SAVE[Save .joblib to MODEL_PATH]
    PKG --> CV[Cross-validation evaluation - variant-specific - see Section 7]
    PKG --> PRED[predict_new_data - impute, predict, expm1, clip, round]
    PRED --> OUT[pred_cluster_{CLUSTER_VALUE}.csv]
```

